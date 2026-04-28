import nki.isa as nisa
import nki.language as nl
from nkilib.core.attention.attention_tkg import attention_tkg as _attention_tkg
from nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.logging import get_logger

_zml_orig_tensor_copy = nisa.tensor_copy


def _zml_tensor_copy_compat(dst, src, engine=None):
    if engine == nisa.scalar_engine:
        return nisa.activation(dst, op=nl.copy, data=src)
    if engine is None:
        return _zml_orig_tensor_copy(dst, src)
    return _zml_orig_tensor_copy(dst, src, engine=engine)


nisa.tensor_copy = _zml_tensor_copy_compat


def decode_tkg(q_hbm, k_active_hbm, v_active, k_prior, v_prior, mask_hbm, attn_mask):
    bs = v_active.shape[0]
    s_active = v_active.shape[2]
    d_head = v_active.shape[3]
    q_head = q_hbm.shape[1] // (bs * s_active)
    full_sprior = k_prior.shape[3]

    q_sb = nl.load(q_hbm)
    k_active_sb = nl.load(k_active_hbm)
    out = nl.ndarray(
        (bs, q_head, d_head, s_active), dtype=q_hbm.dtype, buffer=nl.shared_hbm
    )
    cfg = AttnTKGConfig(
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        curr_sprior=full_sprior,
        full_sprior=full_sprior,
        d_head=d_head,
        block_len=0,
        tp_k_prior=False,
        strided_mm1=True,
        use_pos_id=False,
        fuse_rope=False,
        use_gpsimd_sb2sb=True,
        qk_in_sb=True,
        k_out_in_sb=False,
        out_in_sb=False,
        enable_fa_s_prior_tiling=True,
    )
    sbm = SbufManager(
        0,
        nl.tile_size.total_available_sbuf_size,
        get_logger("decode_tkg"),
        use_auto_alloc=True,
    )
    sbm.open_scope()

    out, _ = _attention_tkg(
        q_sb,
        k_active_sb,
        v_active,
        k_prior,
        v_prior,
        mask_hbm,
        out,
        cfg,
        sbm,
    )
    sbm.close_scope()
    return out


def decode_inhouse(q_hbm, k_active_hbm, v_active, k_prior, v_prior, mask_hbm, attn_mask):
    bs = v_active.shape[0]
    s_active = v_active.shape[2]
    d_head = v_active.shape[3]
    q_head = q_hbm.shape[1] // (bs * s_active)
    full_sprior = k_prior.shape[3]

    assert s_active == 1
    assert k_active_hbm.shape[0] == d_head
    assert v_prior.shape[0] == bs
    assert v_prior.shape[2] == full_sprior
    assert v_prior.shape[3] == d_head
    assert mask_hbm.shape == attn_mask.shape

    out = nl.ndarray(
        (bs, q_head, d_head, s_active), dtype=q_hbm.dtype, buffer=nl.shared_hbm
    )
    qk = nl.ndarray((s_active, full_sprior), dtype=nl.float32, buffer=nl.psum)
    qk_sbuf = nl.ndarray((s_active, full_sprior), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_sbuf = nl.ndarray((d_head, s_active), dtype=out.dtype, buffer=nl.sbuf)

    for bs_i in nl.sequential_range(bs):
        for qh_i in nl.sequential_range(q_head):
            q_offset = bs_i * q_head + qh_i
            q_sbuf = nl.load(q_hbm[:, q_offset : q_offset + 1])
            k_sbuf = nl.load(k_prior[bs_i, 0, :, :])
            v_sbuf = nl.load_transpose2d(v_prior[bs_i, 0, :, :])
            mask_sbuf = nl.load_transpose2d(attn_mask[:, bs_i, qh_i, :])

            nisa.nc_matmul(qk, q_sbuf, k_sbuf)
            nisa.tensor_copy(qk_sbuf, qk)
            qk_sbuf = nl.add(qk_sbuf, mask_sbuf)
            row_max = nl.broadcast_to(
                nl.max(qk_sbuf, axis=1, keepdims=True), qk_sbuf.shape
            )
            norm_row = nl.subtract(qk_sbuf, row_max)
            exp_row = nl.exp(norm_row)
            sum_row = nl.broadcast_to(
                nl.sum(exp_row, axis=1, keepdims=True), exp_row.shape
            )
            scores = nl.multiply(exp_row, nl.reciprocal(sum_row))

            weighted_v = nl.multiply(v_sbuf, nl.broadcast_to(scores, v_sbuf.shape))
            reduced = nl.sum(weighted_v, axis=1, keepdims=True)
            nisa.tensor_copy(out_sbuf, reduced)
            nl.store(dst=out[bs_i, qh_i, :, :], value=out_sbuf)

    return out
