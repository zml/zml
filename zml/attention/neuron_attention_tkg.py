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


def zml_attention_tkg(q_hbm, k_active_hbm, v_active, k_prior, v_prior, mask_hbm):
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
        enable_fa_s_prior_tiling=False,
    )
    sbm = SbufManager(
        0,
        nl.tile_size.total_available_sbuf_size,
        get_logger("zml_attention_tkg"),
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
