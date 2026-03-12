import argparse
import json
import math
import os
import triton

import triton.backends as triton_backends
from triton.backends import Backend as BackendRegistration
from triton.runtime.driver import driver as runtime_driver

from fake_plugin.compiler import Backend as FakeCompilerBackend
from fake_plugin.driver import Driver as FakeDriver
from triton_kernels.unified_attention import (
    kernel_unified_attention_2d_ptr,
    kernel_unified_attention_3d_ptr,
    reduce_segments_ptr,
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


def next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def select_3d_config(
    block_size: int,
    max_seqlen_k: int,
    target_num_prgms: int,
    num_2d_prgms: int,
) -> tuple[dict, dict]:
    reduce_num_warps = 2
    attn_warps = 2
    tile_size = block_size

    num_segments = math.ceil(target_num_prgms / max(1, num_2d_prgms))
    num_segments = next_power_of_2(max(1, num_segments))
    num_segments = min(num_segments, 128)
    min_segments = 16 if tile_size <= 16 else 8
    num_segments = max(num_segments, min_segments)

    if num_segments == min_segments:
        reduce_num_warps = 1

    attn_config = {
        "TILE_SIZE": tile_size,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_warps,
        "num_stages": 1,
    }
    reduce_config = {
        "TILE_SIZE": tile_size,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": reduce_num_warps,
        "num_stages": 1,
    }
    return attn_config, reduce_config


def register_fake_backend() -> None:
    triton_backends.backends["mybackend_runtime"] = BackendRegistration(
        compiler=FakeCompilerBackend,
        driver=FakeDriver,
    )
    runtime_driver.set_active(FakeDriver())


def compile_2d_ptr(cfg: dict) -> str:
    dims = cfg["dimensions"]

    num_tokens = dims["num_tokens"]
    num_heads = dims["num_heads"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    padded_head_dim = next_power_of_2(head_dim)
    num_blocks = dims["num_blocks"]
    block_size = dims["block_size"]
    batch_size = dims["batch_size"]
    num_blocks_per_seq = dims["num_blocks_per_seq"]
    num_qq_tokens = dims.get("num_qq_tokens", 1)
    max_seqlen_q = dims["max_seqlen_q"]

    num_queries_per_kv = num_heads // num_kv_heads

    q_shape = (num_tokens, num_heads, head_dim)
    kv_shape = (num_blocks, block_size, num_kv_heads, head_dim)

    flags = cfg["feature_flags"]
    use_alibi_slopes = flags["use_alibi_slopes"]
    use_softcap = flags["use_softcap"]
    use_sinks = flags["use_sinks"]
    use_fp8 = flags["use_fp8"]
    use_qq_bias = "num_qq_tokens" in dims
    all_decode = flags["all_decode"]
    sliding_window = flags["sliding_window"]

    cfg_2d = cfg["config"]
    block_q = cfg_2d["block_q"]
    block_m = cfg_2d["block_m"]
    tile_size = cfg_2d["tile_size"]
    num_warps = cfg_2d["num_warps"]
    num_stages = cfg_2d["num_stages"]
    total_q_blocks = cfg_2d["total_q_blocks"]

    kwargs = {
        "output_ptr": FakeTensor("bf16", q_shape),
        "query_ptr": FakeTensor("bf16", q_shape),
        "key_cache_ptr": FakeTensor("bf16", kv_shape),
        "value_cache_ptr": FakeTensor("bf16", kv_shape),
        "sink_ptr": FakeTensor("fp32", (num_heads,)),
        "block_tables_ptr": FakeTensor("i32", (batch_size, num_blocks_per_seq)),
        "seq_lens_ptr": FakeTensor("i32", (batch_size,)),
        "alibi_slopes_ptr": FakeTensor("fp32", (num_heads,)),
        "qq_bias_ptr": FakeTensor("fp32", (num_qq_tokens, num_qq_tokens)),
        "scale_ptr": FakeTensor("fp32", (1,)),
        "k_scale_ptr": FakeTensor("fp32", (1,)),
        "v_scale_ptr": FakeTensor("fp32", (1,)),
        "out_scale_ptr": FakeTensor("fp32", (1,)),
        "softcap_ptr": FakeTensor("fp32", (1,)),
        "block_table_stride_ptr": FakeTensor("i64", (1,)),
        "query_stride_0_ptr": FakeTensor("i64", (1,)),
        "query_stride_1_ptr": FakeTensor("i64", (1,)),
        "output_stride_0_ptr": FakeTensor("i64", (1,)),
        "output_stride_1_ptr": FakeTensor("i64", (1,)),
        "qq_bias_stride_0_ptr": FakeTensor("i64", (1,)),
        "stride_k_cache_0_ptr": FakeTensor("i64", (1,)),
        "stride_k_cache_1_ptr": FakeTensor("i64", (1,)),
        "stride_k_cache_2_ptr": FakeTensor("i64", (1,)),
        "stride_v_cache_0_ptr": FakeTensor("i64", (1,)),
        "stride_v_cache_1_ptr": FakeTensor("i64", (1,)),
        "stride_v_cache_2_ptr": FakeTensor("i64", (1,)),
        "query_start_len_ptr": FakeTensor("i32", (batch_size + 1,)),
        "num_seqs_ptr": FakeTensor("i32", (1,)),
        "num_query_heads": num_heads,
        "num_queries_per_kv": num_queries_per_kv,
        "BLOCK_SIZE": block_size,
        "TILE_SIZE": tile_size,
        "HEAD_SIZE": head_dim,
        "HEAD_SIZE_PADDED": padded_head_dim,
        "USE_ALIBI_SLOPES": use_alibi_slopes,
        "USE_QQ_BIAS": use_qq_bias,
        "USE_SOFTCAP": use_softcap,
        "USE_SINKS": use_sinks,
        "SLIDING_WINDOW": sliding_window,
        "stride_k_cache_3": 1,
        "stride_v_cache_3": 1,
        "BLOCK_Q": block_q,
        "BLOCK_M": block_m,
        "USE_FP8": use_fp8,
        "ALL_DECODE": all_decode,
        # runtime options
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    kernel = kernel_unified_attention_2d_ptr.warmup(grid=(num_kv_heads, total_q_blocks), **kwargs)
    return kernel.asm["ttir"]


def compile_3d_ptr(cfg: dict) -> str:
    dims = cfg["dimensions"]

    num_tokens = dims["num_tokens"]
    num_heads = dims["num_heads"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    padded_head_dim = next_power_of_2(head_dim)
    num_blocks = dims["num_blocks"]
    block_size = dims["block_size"]
    batch_size = dims["batch_size"]
    num_blocks_per_seq = dims["num_blocks_per_seq"]
    num_qq_tokens = dims.get("num_qq_tokens", 1)
    cu_count = dims["cu_count"]
    num_queries_per_kv = num_heads // num_kv_heads

    # unused
    max_seqlen_k = 0
    attn_cfg = cfg["config"]

    block_m = attn_cfg["block_m"]
    block_q = attn_cfg["block_q"]
    tile_size = attn_cfg["tile_size"]
    num_warps = attn_cfg["num_warps"]
    num_stages = attn_cfg["num_stages"]
    num_segments_per_seq = attn_cfg["num_segments_per_seq"]
    total_q_blocks = attn_cfg["total_q_blocks"]

    q_shape = (num_tokens, num_heads, head_dim)
    kv_shape = (num_blocks, block_size, num_kv_heads, head_dim)
    segm_out_shape = (num_tokens, num_heads, num_segments_per_seq, triton.next_power_of_2(head_dim))
    segm_stat_shape = (num_tokens, num_heads, num_segments_per_seq)

    flags = cfg["feature_flags"]
    use_alibi_slopes = flags["use_alibi_slopes"]
    use_softcap = flags["use_softcap"]
    use_sinks = flags["use_sinks"]
    all_decode = flags["all_decode"]
    sliding_window = flags["sliding_window"]
    use_qq_bias = "num_qq_tokens" in dims

    kwargs = {
        "query_ptr": FakeTensor("bf16", q_shape),
        "key_cache_ptr": FakeTensor("bf16", kv_shape),
        "value_cache_ptr": FakeTensor("bf16", kv_shape),
        "sink_ptr": FakeTensor("fp32", (num_heads,)),
        "block_tables_ptr": FakeTensor("i32", (batch_size, num_blocks_per_seq)),
        "seq_lens_ptr": FakeTensor("i32", (batch_size,)),
        "alibi_slopes_ptr": FakeTensor("fp32", (num_heads,)),
        "qq_bias_ptr": FakeTensor("fp32", (num_qq_tokens, num_qq_tokens)),
        "scale_ptr": FakeTensor("fp32", (1,)),
        "k_scale_ptr": FakeTensor("fp32", (1,)),
        "v_scale_ptr": FakeTensor("fp32", (1,)),
        "softcap_ptr": FakeTensor("fp32", (1,)),
        "num_query_heads": num_heads,
        "num_queries_per_kv": num_queries_per_kv,
        "block_table_stride_ptr": FakeTensor("i64", (1,)),
        "query_stride_0_ptr": FakeTensor("i64", (1,)),
        "query_stride_1_ptr": FakeTensor("i64", (1,)),
        "qq_bias_stride_0_ptr": FakeTensor("i64", (1,)),
        "BLOCK_SIZE": block_size,
        "TILE_SIZE": tile_size,
        "HEAD_SIZE": head_dim,
        "HEAD_SIZE_PADDED": padded_head_dim,
        "USE_ALIBI_SLOPES": use_alibi_slopes,
        "USE_QQ_BIAS": use_qq_bias,
        "USE_SOFTCAP": use_softcap,
        "USE_SINKS": use_sinks,
        "SLIDING_WINDOW": sliding_window,
        "stride_k_cache_0_ptr": FakeTensor("i64", (1,)),
        "stride_k_cache_1_ptr": FakeTensor("i64", (1,)),
        "stride_k_cache_2_ptr": FakeTensor("i64", (1,)),
        "stride_k_cache_3": 1,
        "stride_v_cache_0_ptr": FakeTensor("i64", (1,)),
        "stride_v_cache_1_ptr": FakeTensor("i64", (1,)),
        "stride_v_cache_2_ptr": FakeTensor("i64", (1,)),
        "stride_v_cache_3": 1,
        "query_start_len_ptr": FakeTensor("i32", (batch_size + 1,)),
        "BLOCK_Q": block_q,
        "num_seqs_ptr": FakeTensor("i32", (1,)),
        "BLOCK_M": block_m,
        "NUM_SEGMENTS_PER_SEQ": num_segments_per_seq,
        "ALL_DECODE": all_decode,
        "segm_output_ptr": FakeTensor("fp32", segm_out_shape),
        "segm_max_ptr": FakeTensor("fp32", segm_stat_shape),
        "segm_expsum_ptr": FakeTensor("fp32", segm_stat_shape),
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    kernel = kernel_unified_attention_3d_ptr.warmup(
        grid=(total_q_blocks, num_kv_heads, num_segments_per_seq),
        **kwargs,
    )
    return kernel.asm["ttir"]


def compile_reduce_ptr(cfg: dict) -> str:
    dims = cfg["dimensions"]

    num_tokens = dims["num_tokens"]
    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    padded_head_dim = next_power_of_2(head_dim)
    batch_size = dims["batch_size"]
    block_size = dims["block_size"]
    num_kv_heads = dims.get("num_kv_heads", 1)
    cu_count = dims["cu_count"]
    num_queries_per_kv = max(1, num_heads // num_kv_heads)

    # unused
    max_seqlen_k = 0


    reduce_cfg = cfg["config"]
    num_segments_per_seq = reduce_cfg["num_segments_per_seq"]
    tile_size = reduce_cfg["tile_size"]
    num_warps = reduce_cfg["num_warps"]
    num_stages = reduce_cfg["num_stages"]
    block_q = reduce_cfg["block_q"]
    
    out_shape = (num_tokens, num_heads, head_dim)
    segm_out_shape = (num_tokens, num_heads, num_segments_per_seq, head_dim)
    segm_stat_shape = (num_tokens, num_heads, num_segments_per_seq)
    out_strides = contiguous_strides(out_shape)

    use_fp8 = cfg["feature_flags"]["use_fp8"]

    kwargs = {
        "segm_output_ptr": FakeTensor("fp32", segm_out_shape),
        "segm_max_ptr": FakeTensor("fp32", segm_stat_shape),
        "segm_expsum_ptr": FakeTensor("fp32", segm_stat_shape),
        "seq_lens_ptr": FakeTensor("i32", (batch_size,)),
        "num_seqs_ptr": FakeTensor("i32", (1,)),
        "num_query_heads": num_heads,
        "out_scale_inv_ptr": FakeTensor("fp32", (1,)),
        "output_stride_0_ptr": FakeTensor("i64", (1,)),
        "output_stride_1_ptr": FakeTensor("i64", (1,)),
        "block_table_stride_ptr": FakeTensor("i64", (1,)),
        "TILE_SIZE": tile_size,
        "HEAD_SIZE": head_dim,
        "HEAD_SIZE_PADDED": padded_head_dim,
        "query_start_len_ptr": FakeTensor("i32", (batch_size + 1,)),
        "BLOCK_Q": block_q,
        "NUM_SEGMENTS_PER_SEQ": num_segments_per_seq,
        "USE_FP8": use_fp8,
        "output_ptr": FakeTensor("bf16", out_shape),
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    kernel = reduce_segments_ptr.warmup(grid=(num_tokens, num_heads), **kwargs)
    return kernel.asm["ttir"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TTIR for wrapped unified-attention kernels")
    parser.add_argument("--config", required=True, help="Raw JSON string")
    args = parser.parse_args()

    cfg = json.loads(args.config)

    register_fake_backend()

    if "kernel_unified_attention_2d_ptr" in cfg:
        ttir = compile_2d_ptr(cfg["kernel_unified_attention_2d_ptr"])
    elif "kernel_unified_attention_3d_ptr" in cfg:
        ttir = compile_3d_ptr(cfg["kernel_unified_attention_3d_ptr"])
    elif "reduce_segments_ptr" in cfg:
        ttir = compile_reduce_ptr(cfg["reduce_segments_ptr"])
    else:
        print("Unknown kernel variant")

    print(ttir)


if __name__ == "__main__":
    main()
