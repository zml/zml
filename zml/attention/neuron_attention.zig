const std = @import("std");

const zml = @import("../zml.zig");
const log = std.log.scoped(.@"zml/attention/neuron");

const cte_source =
    \\try:
    \\    import nki.language as nl
    \\    import nki.isa as nisa
    \\    import nkilib.core.attention.attention_cte as _attention_cte_mod
    \\except ImportError:
    \\    try:
    \\        import nki.language as nl
    \\        import nki.isa as nisa
    \\        import nkilib_standalone.nkilib.core.attention.attention_cte as _attention_cte_mod
    \\    except ImportError:
    \\        import neuronxcc.nki.language as nl
    \\        import neuronxcc.nki.isa as nisa
    \\        import nkilib.core.attention_cte as _attention_cte_mod
    \\
    \\
    \\def _zml_transpose_exp_via_psum(dst, src, src_offset, num_p, num_f, atp, psum):
    \\    num_f_outer = num_f // atp.sb_p
    \\    num_f_inner = num_f % atp.sb_p
    \\
    \\    if num_f_outer >= 1:
    \\        for block_i in range(num_f_outer):
    \\            block_offset = src_offset + block_i * atp.sb_p
    \\            nisa.nc_transpose(
    \\                psum[: atp.sb_p, :num_p],
    \\                src[:num_p, nl.ds(block_offset, atp.sb_p)],
    \\            )
    \\            nisa.tensor_copy(
    \\                dst[: atp.sb_p, nl.ds(block_i * atp.sb_p, num_p)],
    \\                psum[: atp.sb_p, :num_p],
    \\            )
    \\
    \\    if num_f_inner > 0:
    \\        block_offset = src_offset + num_f_outer * atp.sb_p
    \\        nisa.nc_transpose(
    \\            psum[:num_f_inner, :num_p],
    \\            src[:num_p, nl.ds(block_offset, num_f_inner)],
    \\        )
    \\        nisa.tensor_copy(
    \\            dst[:num_f_inner, nl.ds(num_f_outer * atp.sb_p, num_p)],
    \\            psum[:num_f_inner, :num_p],
    \\        )
    \\
    \\
    \\def _zml_patched_exp_impl(grp_i, ac, atp, sp, bufs, sink):
    \\    has_any_compute_pred = (
    \\        _attention_cte_mod._has_any_compute_causal(grp_i, sp.section_offset_active, ac)
    \\        if (atp.is_causal and not sp.section_contains_prefix)
    \\        else True
    \\    )
    \\    if not has_any_compute_pred:
    \\        return
    \\
    \\    q_seqlen_offset = grp_i * atp.sb_p
    \\    nisa.memset(bufs.exp_partial_sum[grp_i][...], value=0.0)
    \\    for large_tile_idx in range(atp.num_large_tiles_per_section):
    \\        _attention_cte_mod.kernel_assert(
    \\            atp.exp_inst_elems == 512, "Internal validation failed."
    \\        )
    \\
    \\        for exp_tile_idx in range(atp.num_exp_insts_per_large_tile):
    \\            is_prior_tile, seqlen_k, k_start_pos, _ = _attention_cte_mod._get_kv_tile_apc(
    \\                ac.is_prefix_caching,
    \\                False,
    \\                True,
    \\                atp.seqlen_k_active_updated,
    \\                ac.seqlen_k_prior,
    \\                sp.section_offset + large_tile_idx * _attention_cte_mod._LARGE_TILE_SZ + exp_tile_idx * atp.exp_inst_elems,
    \\                None,
    \\            )
    \\            num_p = min(ac.seqlen_q - q_seqlen_offset, _attention_cte_mod._Q_GRP_SZ)
    \\            num_f = min(seqlen_k - k_start_pos, atp.exp_inst_elems)
    \\
    \\            if atp.is_causal and not is_prior_tile:
    \\                exp_sel_mask = _attention_cte_mod._has_any_compute_causal(grp_i, k_start_pos, ac)
    \\            else:
    \\                exp_sel_mask = True
    \\
    \\            if ac.use_swa and atp.is_causal and not is_prior_tile:
    \\                exp_sel_mask = exp_sel_mask and _attention_cte_mod._has_any_compute_swa(
    \\                    grp_i, k_start_pos, atp.exp_inst_elems, ac
    \\                )
    \\
    \\            if exp_sel_mask and seqlen_k > k_start_pos:
    \\                nisa.activation_reduce(
    \\                    bufs.exp_sb[grp_i][large_tile_idx][:num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)],
    \\                    op=nl.exp,
    \\                    data=bufs.mm1_masked[grp_i][large_tile_idx][
    \\                        :num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)
    \\                    ],
    \\                    reduce_op=nl.add,
    \\                    reduce_res=bufs.exp_partial_sum[grp_i][
    \\                        :num_p,
    \\                        large_tile_idx * atp.num_exp_insts_per_large_tile + exp_tile_idx,
    \\                    ],
    \\                    bias=bufs.mm1_running_max[:num_p, grp_i],
    \\                )
    \\
    \\                _zml_transpose_exp_via_psum(
    \\                    dst=bufs.exp_tp_sb[grp_i][large_tile_idx][exp_tile_idx],
    \\                    src=bufs.exp_sb[grp_i][large_tile_idx],
    \\                    src_offset=exp_tile_idx * atp.mm2_grp_sz,
    \\                    num_p=num_p,
    \\                    num_f=num_f,
    \\                    atp=atp,
    \\                    psum=bufs.mm1_psum[grp_i][large_tile_idx][exp_tile_idx],
    \\                )
    \\
    \\    if (sink is not None) and (sp.section_idx == 0):
    \\        frs_sink_idx = bufs.exp_partial_sum[grp_i].shape[-1] - 1
    \\        nisa.activation(
    \\            bufs.exp_partial_sum[grp_i][:, frs_sink_idx],
    \\            op=nl.exp,
    \\            data=bufs.sink_sb,
    \\            bias=bufs.mm1_running_max[:, grp_i],
    \\        )
    \\
    \\
    \\def _zml_serial_attention_cte_impl(
    \\    q,
    \\    k_active,
    \\    v_active,
    \\    k_prior,
    \\    v_prior,
    \\    prior_used_len,
    \\    o,
    \\    batch_id,
    \\    batch_id_kv,
    \\    ac,
    \\    sink=None,
    \\    out_neg_max=None,
    \\    out_sum_recip=None,
    \\    shard_seqlen_q_start=-1,
    \\    shard_seqlen_q_length=-1,
    \\    cp_offset=None,
    \\    bound_min=None,
    \\    bound_max=None,
    \\):
    \\    is_seqlen_sharded = shard_seqlen_q_start >= 0
    \\    if is_seqlen_sharded and shard_seqlen_q_length <= 1:
    \\        return _zml_orig_attention_cte_impl(
    \\            q,
    \\            k_active,
    \\            v_active,
    \\            k_prior,
    \\            v_prior,
    \\            prior_used_len,
    \\            o,
    \\            batch_id,
    \\            batch_id_kv,
    \\            ac,
    \\            sink=sink,
    \\            out_neg_max=out_neg_max,
    \\            out_sum_recip=out_sum_recip,
    \\            shard_seqlen_q_start=shard_seqlen_q_start,
    \\            shard_seqlen_q_length=shard_seqlen_q_length,
    \\            cp_offset=cp_offset,
    \\            bound_min=bound_min,
    \\            bound_max=bound_max,
    \\        )
    \\
    \\    atp = _attention_cte_mod._compute_tile_parameters(ac, is_seqlen_sharded)
    \\    start = shard_seqlen_q_start if is_seqlen_sharded else 0
    \\    length = shard_seqlen_q_length if is_seqlen_sharded else atp.num_grps
    \\
    \\    for grp_i in range(start, start + length):
    \\        _zml_orig_attention_cte_impl(
    \\            q,
    \\            k_active,
    \\            v_active,
    \\            k_prior,
    \\            v_prior,
    \\            prior_used_len,
    \\            o,
    \\            batch_id,
    \\            batch_id_kv,
    \\            ac,
    \\            sink=sink,
    \\            out_neg_max=out_neg_max,
    \\            out_sum_recip=out_sum_recip,
    \\            shard_seqlen_q_start=grp_i,
    \\            shard_seqlen_q_length=1,
    \\            cp_offset=cp_offset,
    \\            bound_min=bound_min,
    \\            bound_max=bound_max,
    \\        )
    \\
    \\
    \\_zml_orig_attention_cte_impl = _attention_cte_mod._attention_cte_impl
    \\_attention_cte_mod._exp_impl = _zml_patched_exp_impl
    \\_attention_cte_mod._attention_cte_impl = _zml_serial_attention_cte_impl
    \\_attention_cte = _attention_cte_mod.attention_cte
    \\
    \\
    \\def zml_attention_cte(q, k, v):
    \\    return _attention_cte(
    \\        q,
    \\        k,
    \\        v,
    \\        scale=1.0,
    \\        causal_mask=True,
    \\        tp_q=False,
    \\        tp_k=False,
    \\        tp_out=False,
    \\    )
;

const prefill_source =
    \\try:
    \\    import nki.language as nl
    \\    import nki.isa as nisa
    \\except ImportError:
    \\    import neuronxcc.nki.language as nl
    \\    import neuronxcc.nki.isa as nisa
    \\
    \\_FLOAT32_MIN = -3.4028235e38
    \\
    \\
    \\def zml_attention_prefill(q, k, v):
    \\    bs, d, seqlen_q = q.shape
    \\    _, _, seqlen_k = k.shape
    \\    assert seqlen_q == seqlen_k
    \\    assert seqlen_q <= nl.tile_size.pmax
    \\    assert seqlen_k <= nl.tile_size.pmax
    \\    assert d <= nl.tile_size.pmax
    \\
    \\    out = nl.ndarray((bs, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)
    \\
    \\    q_sb = nl.ndarray((d, nl.tile_size.pmax), dtype=q.dtype, buffer=nl.sbuf)
    \\    k_sb = nl.ndarray((d, nl.tile_size.pmax), dtype=k.dtype, buffer=nl.sbuf)
    \\    v_sb = nl.ndarray((nl.tile_size.pmax, d), dtype=v.dtype, buffer=nl.sbuf)
    \\    scores_psum = nl.ndarray((nl.tile_size.pmax, nl.tile_size.pmax), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
    \\    out_psum = nl.ndarray((nl.tile_size.pmax, d), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
    \\    scores_sb = nl.ndarray((nl.tile_size.pmax, nl.tile_size.pmax), dtype=nl.float32, buffer=nl.sbuf)
    \\    masked_sb = nl.ndarray((nl.tile_size.pmax, nl.tile_size.pmax), dtype=nl.float32, buffer=nl.sbuf)
    \\    exp_sb = nl.ndarray((nl.tile_size.pmax, nl.tile_size.pmax), dtype=q.dtype, buffer=nl.sbuf)
    \\    exp_t_sb = nl.ndarray((nl.tile_size.pmax, nl.tile_size.pmax), dtype=q.dtype, buffer=nl.sbuf)
    \\    out_sb = nl.ndarray((nl.tile_size.pmax, d), dtype=q.dtype, buffer=nl.sbuf)
    \\    row_max = nl.ndarray((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    \\    row_sum = nl.ndarray((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    \\    zero_bias = nl.ndarray((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    \\    nisa.memset(zero_bias, value=0.0)
    \\
    \\    for batch_idx in range(bs):
    \\        nisa.dma_copy(
    \\            dst=q_sb.ap(pattern=[[seqlen_q, d], [1, seqlen_q]], offset=0),
    \\            src=q.ap(pattern=[[seqlen_q, d], [1, seqlen_q]], offset=batch_idx * d * seqlen_q),
    \\        )
    \\        nisa.dma_copy(
    \\            dst=k_sb.ap(pattern=[[seqlen_k, d], [1, seqlen_k]], offset=0),
    \\            src=k.ap(pattern=[[seqlen_k, d], [1, seqlen_k]], offset=batch_idx * d * seqlen_k),
    \\        )
    \\        nisa.dma_copy(
    \\            dst=v_sb.ap(pattern=[[d, seqlen_k], [1, d]], offset=0),
    \\            src=v.ap(pattern=[[d, seqlen_k], [1, d]], offset=batch_idx * seqlen_k * d),
    \\        )
    \\
    \\        nisa.nc_matmul(
    \\            scores_psum[:seqlen_q, :seqlen_k],
    \\            q_sb[:d, :seqlen_q],
    \\            k_sb[:d, :seqlen_k],
    \\        )
    \\        nisa.tensor_copy(scores_sb[:seqlen_q, :seqlen_k], scores_psum[:seqlen_q, :seqlen_k])
    \\
    \\        nisa.affine_select(
    \\            masked_sb[:seqlen_q, :seqlen_k],
    \\            pattern=[[-1, seqlen_k]],
    \\            offset=0,
    \\            channel_multiplier=1,
    \\            cmp_op=nl.greater_equal,
    \\            on_true_tile=scores_sb[:seqlen_q, :seqlen_k],
    \\            on_false_value=_FLOAT32_MIN,
    \\        )
    \\
    \\        nisa.tensor_reduce(
    \\            row_max[:seqlen_q, :],
    \\            op=nl.maximum,
    \\            data=masked_sb[:seqlen_q, :seqlen_k],
    \\            axis=1,
    \\            keepdims=True,
    \\        )
    \\        nisa.tensor_scalar(row_max[:seqlen_q, :], row_max[:seqlen_q, :], nl.multiply, -1.0)
    \\        nisa.activation(
    \\            scores_sb[:seqlen_q, :seqlen_k],
    \\            op=nl.copy,
    \\            data=masked_sb[:seqlen_q, :seqlen_k],
    \\            scale=1.0,
    \\            bias=row_max[:seqlen_q],
    \\        )
    \\        nisa.memset(row_sum[:seqlen_q, :], value=0.0)
    \\        nisa.activation_reduce(
    \\            exp_sb[:seqlen_q, :seqlen_k],
    \\            op=nl.exp,
    \\            data=scores_sb[:seqlen_q, :seqlen_k],
    \\            reduce_op=nl.add,
    \\            reduce_res=row_sum[:seqlen_q, :],
    \\            bias=zero_bias[:seqlen_q, :],
    \\        )
    \\        nisa.reciprocal(row_sum[:seqlen_q, :], row_sum[:seqlen_q, :])
    \\
    \\        nisa.nc_transpose(scores_psum[:seqlen_k, :seqlen_q], exp_sb[:seqlen_q, :seqlen_k])
    \\        nisa.tensor_copy(exp_t_sb[:seqlen_k, :seqlen_q], scores_psum[:seqlen_k, :seqlen_q])
    \\        nisa.nc_matmul(
    \\            out_psum[:seqlen_q, :d],
    \\            exp_t_sb[:seqlen_k, :seqlen_q],
    \\            v_sb[:seqlen_k, :d],
    \\        )
    \\        nisa.activation(
    \\            out_sb[:seqlen_q, :d],
    \\            op=nl.copy,
    \\            data=out_psum[:seqlen_q, :d],
    \\            scale=row_sum[:seqlen_q],
    \\            bias=zero_bias[:seqlen_q],
    \\        )
    \\        nisa.dma_copy(
    \\            dst=out.ap(pattern=[[d, seqlen_q], [1, d]], offset=batch_idx * seqlen_q * d),
    \\            src=out_sb.ap(pattern=[[d, seqlen_q], [1, d]], offset=0),
    \\        )
    \\
    \\    return out
;

const tkg_source =
    \\try:
    \\    import nki.language as nl
    \\    from nkilib.core.attention.attention_tkg import attention_tkg as _attention_tkg
    \\    from nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
    \\    from nkilib.core.utils.allocator import SbufManager
    \\    from nkilib.core.utils.logging import get_logger
    \\except ImportError:
    \\    try:
    \\        import nki.language as nl
    \\        from nkilib_standalone.nkilib.core.attention.attention_tkg import attention_tkg as _attention_tkg
    \\        from nkilib_standalone.nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
    \\        from nkilib_standalone.nkilib.core.utils.allocator import SbufManager
    \\        from nkilib_standalone.nkilib.core.utils.logging import get_logger
    \\    except ImportError:
    \\        import neuronxcc.nki.language as nl
    \\        from nkilib.core.attention_tkg import attention_tkg as _attention_tkg, AttnTKGConfig
    \\        from nkilib.core.utils.allocator import SbufManager
    \\        from nkilib.core.utils.logging import get_logger
    \\
    \\
    \\def zml_attention_tkg(q_hbm, k_active_hbm, v_active, k_prior, v_prior, mask_hbm, active_blocks_table):
    \\    bs = v_active.shape[0]
    \\    s_active = v_active.shape[2]
    \\    d_head = v_active.shape[3]
    \\    q_head = q_hbm.shape[1] // (bs * s_active)
    \\    block_len = k_prior.shape[1]
    \\    num_blocks = active_blocks_table.shape[1]
    \\    full_sprior = num_blocks * block_len
    \\
    \\    q_sb = nl.load(q_hbm)
    \\    k_active_sb = nl.load(k_active_hbm)
    \\    out = nl.ndarray((bs, q_head, d_head, s_active), dtype=q_hbm.dtype, buffer=nl.shared_hbm)
    \\
    \\    cfg = AttnTKGConfig(
    \\        bs=bs,
    \\        q_head=q_head,
    \\        s_active=s_active,
    \\        curr_sprior=full_sprior,
    \\        full_sprior=full_sprior,
    \\        d_head=d_head,
    \\        block_len=block_len,
    \\        tp_k_prior=True,
    \\        strided_mm1=False,
    \\        use_pos_id=False,
    \\        fuse_rope=False,
    \\        use_gpsimd_sb2sb=True,
    \\        qk_in_sb=True,
    \\        k_out_in_sb=False,
    \\        out_in_sb=False,
    \\        enable_fa_s_prior_tiling=True,
    \\    )
    \\    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, get_logger("zml_attention_tkg"), use_auto_alloc=True)
    \\    sbm.open_scope()
    \\
    \\    out, _ = _attention_tkg(
    \\        q_sb,
    \\        k_active_sb,
    \\        v_active,
    \\        k_prior,
    \\        v_prior,
    \\        mask_hbm,
    \\        out,
    \\        cfg,
    \\        sbm,
    \\        active_blocks_table=active_blocks_table,
    \\    )
    \\    sbm.close_scope()
    \\    return out
;

fn vanillaFallback(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const seq_len = k.dim(.k);
    var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, q.dtype(), null);
    attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = q.dim(.q) }, attn_mask.dtype()), token_index.reshape(.{ .coord = 1 }), .{});
    return zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true });
}

fn canUseAttentionCte(q: zml.Tensor, k: zml.Tensor, token_index: zml.Tensor) bool {
    _ = token_index;
    if (q.dim(.q) <= 1) return false;
    if (q.dim(.q) > 128) return false;
    if (q.dim(.q) != k.dim(.k)) return false;
    if (q.dim(.hd) > 128) return false;

    return switch (q.dtype()) {
        .bf16 => true,
        else => false,
    };
}

fn chooseBlockLen(full_sprior: i64) ?i64 {
    inline for ([_]i64{ 128, 64, 32, 16, 8 }) |candidate| {
        if (@mod(full_sprior, candidate) == 0) return candidate;
    }
    return null;
}

fn canUseAttentionTkg(q: zml.Tensor, k: zml.Tensor, token_index: zml.Tensor) bool {
    if (q.dim(.q) != 1) return false;
    if (q.dim(.hd) > 128) return false;
    if (token_index.rank() != 0) return false;
    if (@mod(q.dim(.h), k.dim(.h)) != 0) return false;
    if (chooseBlockLen(k.dim(.k)) == null) return false;

    return switch (q.dtype()) {
        .bf16, .f16, .f32 => true,
        else => false,
    };
}

fn prefillAttentionCte(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
    _ = token_index;

    if (q.shape().hasTag(.b) != null) {
        const batch_size = q.dim(.b);
        const num_heads = q.dim(.h);

        const q_cte = q.scale(scale)
            .transpose(.{ .b, .h, .hd, .q })
            .merge(.{ .bh = .{ .b, .h } })
            .rename(.{ .bh = .b, .hd = .d, .q = .s });
        const k_cte = k.transpose(.{ .b, .h, .hd, .k })
            .merge(.{ .bh = .{ .b, .h } })
            .rename(.{ .bh = .b, .hd = .d });
        const v_cte = v.transpose(.{ .b, .h, .k, .hd })
            .merge(.{ .bh = .{ .b, .h } })
            .rename(.{ .bh = .b, .hd = .d });

        var out = zml.ops.neuronNki(
            .{ q_cte, k_cte, v_cte },
            .{ zml.Shape.init(.{ .b = q_cte.dim(.b), .s = q_cte.dim(.s), .d = q_cte.dim(.d) }, q_cte.dtype()) },
            .{
                .name = "attention_cte",
                .entrypoint = "zml_attention_prefill",
                .source = prefill_source,
            },
        )[0];

        out = out.rename(.{ .b = .bh, .s = .q, .d = .hd }).splitAxis(.bh, .{ .b = batch_size, .h = num_heads }).transpose(.{ .b, .q, .h, .hd });
        return out;
    }

    const num_heads = q.dim(.h);
    const q_cte = q.scale(scale).transpose(.{ .h, .hd, .q }).rename(.{ .h = .b, .hd = .d, .q = .s });
    const k_cte = k.transpose(.{ .h, .hd, .k }).rename(.{ .h = .b, .hd = .d });
    const v_cte = v.transpose(.{ .h, .k, .hd }).rename(.{ .h = .b, .hd = .d });

    var out = zml.ops.neuronNki(
        .{ q_cte, k_cte, v_cte },
        .{ zml.Shape.init(.{ .b = q_cte.dim(.b), .s = q_cte.dim(.s), .d = q_cte.dim(.d) }, q_cte.dtype()) },
        .{
            .name = "attention_cte",
            .entrypoint = "zml_attention_prefill",
            .source = prefill_source,
        },
    )[0];

    out = out.rename(.{ .b = .h, .s = .q, .d = .hd }).transpose(.{ .q, .h, .hd });
    std.debug.assert(out.dim(.h) == num_heads);
    return out;
}

fn decodeAttentionTkg(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const block_len = chooseBlockLen(k.dim(.k)).?;
    const had_batch = q.shape().hasTag(.b) != null;
    const q_batched = if (had_batch) q else q.insertAxes(.q, .{.b});
    const k_batched = if (had_batch) k else k.insertAxes(.k, .{.b});
    const v_batched = if (had_batch) v else v.insertAxes(.k, .{.b});

    const batch_size = q_batched.dim(.b);
    const num_heads = q_batched.dim(.h);
    const num_kv_heads = k_batched.dim(.h);
    const heads_per_kv = @divExact(num_heads, num_kv_heads);
    const head_dim = q_batched.dim(.hd);
    const full_sprior = k_batched.dim(.k);
    const num_blocks = @divExact(full_sprior, block_len);
    const effective_batch = batch_size * num_kv_heads;
    const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(head_dim))));
    const kernel_dtype: zml.DataType = .f32;

    const q_grouped = q_batched.convert(kernel_dtype).splitAxis(.h, .{ .kvh = num_kv_heads, .hq = .auto });
    const q_tkg = q_grouped.scale(scale)
        .transpose(.{ .hd, .b, .kvh, .hq, .q })
        .merge(.{ .beff = .{ .b, .kvh } })
        .merge(.{ .tot = .{ .beff, .hq, .q } })
        .rename(.{ .hd = .d });

    const active_k = k_batched.convert(kernel_dtype).dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = token_index, .len = 1 } });
    const active_v = v_batched.convert(kernel_dtype).dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = token_index, .len = 1 } });

    const k_active_tkg = active_k.transpose(.{ .hd, .b, .h, .k })
        .merge(.{ .tot = .{ .b, .h, .k } })
        .rename(.{ .hd = .d });
    const v_active_tkg = active_v.transpose(.{ .b, .h, .k, .hd })
        .merge(.{ .beff = .{ .b, .h } })
        .rename(.{ .beff = .b, .hd = .d })
        .insertAxes(.k, .{.one});

    const k_prior_tkg = k_batched.convert(kernel_dtype).transpose(.{ .b, .h, .k, .hd })
        .merge(.{ .beff = .{ .b, .h } })
        .splitAxis(.k, .{ .blk = num_blocks, .blen = block_len })
        .transpose(.{ .beff, .blk, .blen, .hd })
        .merge(.{ .row = .{ .beff, .blk } });
    const v_prior_tkg = v_batched.convert(kernel_dtype).transpose(.{ .b, .h, .k, .hd })
        .merge(.{ .beff = .{ .b, .h } })
        .splitAxis(.k, .{ .blk = num_blocks, .blen = block_len })
        .transpose(.{ .beff, .blk, .blen, .hd })
        .merge(.{ .row = .{ .beff, .blk } });

    const block_ids = zml.Tensor.iota(zml.Shape.init(.{ .blk = num_blocks }, .i32), .blk).convert(.u32);
    const active_blocks_table = block_ids.insertAxes(.blk, .{.b}).broad(zml.Shape.init(.{ .b = effective_batch, .blk = num_blocks }, .u32));

    const mask_shape_dtype = zml.Shape.init(.{ .k = full_sprior, .b = effective_batch, .hq = heads_per_kv, .q = 1 }, token_index.dtype());
    const mask_shape_u8 = zml.Shape.init(.{ .k = full_sprior, .b = effective_batch, .hq = heads_per_kv, .q = 1 }, .u8);
    const prior_positions = zml.Tensor.iota(zml.Shape.init(.{ .k = full_sprior }, .i32), .k).convert(token_index.dtype());
    const prior_mask = prior_positions
        .insertAxes(.last, .{ .b, .hq, .q })
        .broad(mask_shape_dtype)
        .cmp(.LT, token_index);
    const mask = prior_mask.select(
        zml.Tensor.constant(.{ .u8 = 1 }).broad(mask_shape_u8),
        zml.Tensor.constant(.{ .u8 = 0 }).broad(mask_shape_u8),
    );

    var out = zml.ops.neuronNki(
        .{ q_tkg, k_active_tkg, v_active_tkg, k_prior_tkg, v_prior_tkg, mask, active_blocks_table },
        .{ zml.Shape.init(.{ .b = effective_batch, .hq = heads_per_kv, .d = head_dim, .q = 1 }, kernel_dtype) },
        .{
            .name = "attention_tkg",
            .entrypoint = "zml_attention_tkg",
            .source = tkg_source,
        },
    )[0];

    out = out
        .splitAxis(.b, .{ .b = batch_size, .kvh = num_kv_heads })
        .merge(.{ .h = .{ .kvh, .hq } })
        .rename(.{ .d = .hd })
        .transpose(.{ .b, .q, .h, .hd })
        .convert(q.dtype());

    return if (had_batch) out else out.squeeze(.b);
}

pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    if (canUseAttentionTkg(q, k, token_index)) {
        const had_batch = q.shape().hasTag(.b) != null;
        const block_len = chooseBlockLen(k.dim(.k)).?;
        log.warn("using neuron decode kernel attention_tkg batched={} synthetic_batch={} q={} k={} h={} kv_h={} hd={} block_len={}", .{
            had_batch,
            !had_batch,
            q.dim(.q),
            k.dim(.k),
            q.dim(.h),
            k.dim(.h),
            q.dim(.hd),
            block_len,
        });
        return decodeAttentionTkg(q, k, v, token_index);
    }
    if (canUseAttentionCte(q, k, token_index)) {
        const had_batch = q.shape().hasTag(.b) != null;
        log.warn("using neuron prefill kernel attention_prefill batched={} synthetic_batch=false q={} k={} h={} kv_h={} hd={}", .{
            had_batch,
            q.dim(.q),
            k.dim(.k),
            q.dim(.h),
            k.dim(.h),
            q.dim(.hd),
        });
        return prefillAttentionCte(q, k, v, token_index);
    }
    log.warn("falling back to vanilla attention on neuron batched={} q={} k={} h={} kv_h={} hd={}", .{
        q.shape().hasTag(.b) != null,
        q.dim(.q),
        k.dim(.k),
        q.dim(.h),
        k.dim(.h),
        q.dim(.hd),
    });
    return vanillaFallback(q, k, v, token_index);
}
