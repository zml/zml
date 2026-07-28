import nki

from nkilib_runtime.core.attention.attention_segmented_cte import attention_segmented_cte
from nkilib_runtime.experimental.transformer.attention_block_tkg import attention_block_tkg
from nkilib_runtime.core.utils.common_types import QuantizationType


@nki.jit
def paged_attention_decode(
    x,
    w_qkv,
    k_cache,
    v_cache,
    active_blocks_table,
    active_mask,
    pos_ids,
    cache_update_idx,
    w_out,
    cos,
    sin,
):
    """Run the fused decode kernel selected by vLLM Neuron 0.21."""
    return attention_block_tkg(
        X=x,
        X_hidden_dim_actual=x.shape[-1],
        rmsnorm_X_enabled=False,
        rmsnorm_X_eps=None,
        rmsnorm_X_gamma=None,
        W_qkv=w_qkv,
        bias_qkv=None,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=1e-6,
        rmsnorm_QK_pre_rope_W_Q=None,
        rmsnorm_QK_pre_rope_W_K=None,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=1e-6,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=active_blocks_table,
        K_cache=k_cache,
        V_cache=v_cache,
        attention_mask=active_mask,
        sink=None,
        update_cache=True,
        kv_cache_update_idx=cache_update_idx,
        W_out=w_out,
        bias_out=None,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=False,
        pos_ids=pos_ids,
    )


@nki.jit
def paged_attention_prefill(
    q,
    k_cache,
    v_cache,
    block_tables,
    prior_tokens,
):
    """Run segmented paged attention with Q pre-scaled by the caller."""
    block_size = v_cache.shape[2]
    num_q_heads = q.shape[0] // block_tables.shape[0]
    return attention_segmented_cte(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        prior_tokens=prior_tokens,
        block_size=block_size,
        prior_seg_size=512,
        num_q_heads=num_q_heads,
    )
