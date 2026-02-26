import os
import torch

from wrap_3d_unified_attention import run_3d_unified_attention_kernels


torch.set_printoptions(threshold=torch.inf)
torch.manual_seed(0)

os.environ["TRITON_BACKEND_DEBUG"] = "1"
os.environ["SHOULD_LOG"] = "1"

token_count = 8
batch_size = 8
num_heads = 32
num_kv_heads = 8
head_size = 128
head_size_padded = 1 << (head_size - 1).bit_length()
block_size = 16
num_blocks = 4096
max_input_len = 1
max_seq_len = block_size
scale = 0.08838834765
alibi_slopes = None
k_scale = 1.0
v_scale = 1.0


def generate_3d():
    query = torch.zeros(
        token_count, num_heads, head_size, dtype=torch.bfloat16, device="cuda"
    )
    k = torch.zeros(
        token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda"
    )
    v = torch.zeros(
        token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda"
    )
    o = torch.empty_like(query)
    key_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    block_tables = torch.zeros(
        (batch_size, max_seq_len // block_size), dtype=torch.int32, device="cuda"
    )
    context_seq_lens = torch.full(
        (batch_size,), max_input_len, dtype=torch.int32, device="cuda"
    )
    start_loc = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device="cuda"
    )

    num_par_softmax_segments = 8
    softmax_segm_output = torch.empty(
        (token_count, num_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
        device="cuda",
    )
    softmax_segm_max = torch.empty(
        (token_count, num_heads, num_par_softmax_segments),
        dtype=torch.float32,
        device="cuda",
    )
    softmax_segm_expsum = torch.empty(
        (token_count, num_heads, num_par_softmax_segments),
        dtype=torch.float32,
        device="cuda",
    )

    print("Parameters for 3D unified attention:")
    print(f"query shape: {query.shape}, dtype: {query.dtype}")
    print(f"k shape: {k.shape}, dtype: {k.dtype}")
    print(f"v shape: {v.shape}, dtype: {v.dtype}")
    print(f"o shape: {o.shape}, dtype: {o.dtype}")
    print(f"key_cache shape: {key_cache.shape}, dtype: {key_cache.dtype}")
    print(f"value_cache shape: {value_cache.shape}, dtype: {value_cache.dtype}")
    print(f"block_tables shape: {block_tables.shape}, dtype: {block_tables.dtype}")
    print(f"start_loc shape: {start_loc.shape}, dtype: {start_loc.dtype}")
    print(f"context_seq_lens shape: {context_seq_lens.shape}, dtype: {context_seq_lens.dtype}")
    print(f"max_input_len: {max_input_len}")
    print(f"k_scale: {k_scale}")
    print(f"v_scale: {v_scale}")
    print(f"alibi_slopes: {alibi_slopes}")
    print(f"scale: {scale}")

    run_3d_unified_attention_kernels(
        q=query,
        k=key_cache,
        v=value_cache,
        out=o,
        cu_seqlens_q=start_loc,
        max_seqlen_q=max_input_len,
        seqused_k=context_seq_lens,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=True,
        window_size=[-1, -1],
        block_table=block_tables,
        softcap=0.0,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    print(o[:, 0, 0])


if __name__ == "__main__":
    generate_3d()
