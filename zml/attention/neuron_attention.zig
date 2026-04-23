const std = @import("std");

const zml = @import("../zml.zig");

const log = std.log.scoped(.@"zml/attention/neuron");

const tkg_source =
    \\try:
    \\    import nki.language as nl
    \\    import nki.isa as nisa
    \\    from nkilib.core.attention.attention_tkg import attention_tkg as _attention_tkg
    \\    from nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
    \\    from nkilib.core.utils.allocator import SbufManager
    \\    from nkilib.core.utils.logging import get_logger
    \\except ImportError:
    \\    try:
    \\        import nki.language as nl
    \\        import nki.isa as nisa
    \\        from nkilib_standalone.nkilib.core.attention.attention_tkg import attention_tkg as _attention_tkg
    \\        from nkilib_standalone.nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
    \\        from nkilib_standalone.nkilib.core.utils.allocator import SbufManager
    \\        from nkilib_standalone.nkilib.core.utils.logging import get_logger
    \\    except ImportError:
    \\        import neuronxcc.nki.language as nl
    \\        import neuronxcc.nki.isa as nisa
    \\        from nkilib.core.attention_tkg import attention_tkg as _attention_tkg, AttnTKGConfig
    \\        from nkilib.core.utils.allocator import SbufManager
    \\        from nkilib.core.utils.logging import get_logger
    \\
    \\_zml_orig_tensor_copy = nisa.tensor_copy
    \\
    \\
    \\def _zml_tensor_copy_compat(dst, src, engine=None):
    \\    if engine == nisa.scalar_engine:
    \\        return nisa.activation(dst, op=nl.copy, data=src)
    \\    if engine == None:
    \\        return _zml_orig_tensor_copy(dst, src)
    \\    return _zml_orig_tensor_copy(dst, src, engine=engine)
    \\
    \\
    \\nisa.tensor_copy = _zml_tensor_copy_compat
    \\
    \\
    \\def zml_attention_tkg(q_hbm, k_active_hbm, v_active, k_prior, v_prior, mask_hbm):
    \\    bs = v_active.shape[0]
    \\    s_active = v_active.shape[2]
    \\    d_head = v_active.shape[3]
    \\    q_head = q_hbm.shape[1] // (bs * s_active)
    \\    full_sprior = k_prior.shape[3]
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
    \\        block_len=0,
    \\        tp_k_prior=False,
    \\        strided_mm1=True,
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
    \\    )
    \\    sbm.close_scope()
    \\    return out
;

fn alignDecodeSprior(full_sprior: i64) i64 {
    return std.mem.alignForward(i64, full_sprior, 256);
}

const BatchTag = enum {
    none,
    short,
    long,
};

fn batchTagOf(t: zml.Tensor) BatchTag {
    if (t.shape().hasTag(.b) != null) return .short;
    if (t.shape().hasTag(.batch) != null) return .long;
    return .none;
}

fn normalizeBatchTag(t: zml.Tensor) zml.Tensor {
    return switch (batchTagOf(t)) {
        .long => t.rename(.{ .batch = .b }),
        else => t,
    };
}

fn restoreBatchTag(t: zml.Tensor, batch_tag: BatchTag) zml.Tensor {
    return switch (batch_tag) {
        .long => t.rename(.{ .b = .batch }),
        else => t,
    };
}

fn normalizeTokenIndex(token_index: zml.Tensor) zml.Tensor {
    return switch (batchTagOf(token_index)) {
        .none => token_index,
        .short => token_index.squeeze(.b),
        .long => token_index.squeeze(.batch),
    };
}

fn canUseAttentionTkg(q: zml.Tensor, k: zml.Tensor, token_index: zml.Tensor) bool {
    if (q.dim(.q) != 1) return false;
    if (q.dim(.hd) > 128) return false;
    if (@mod(q.dim(.h), k.dim(.h)) != 0) return false;

    const token_index_scalar = normalizeTokenIndex(token_index);
    if (token_index_scalar.rank() != 0) return false;

    return switch (q.dtype()) {
        .bf16, .f16, .f32 => true,
        else => false,
    };
}

fn decodeAttentionTkgUnchunked(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const batch_tag = batchTagOf(q);
    const token_index_scalar = normalizeTokenIndex(token_index);
    const q_norm = normalizeBatchTag(q);
    const k_norm = normalizeBatchTag(k);
    const v_norm = normalizeBatchTag(v);
    const had_batch = batch_tag != .none;
    const q_batched = if (had_batch) q_norm else q_norm.insertAxes(.q, .{.b});
    const k_unpadded = if (had_batch) k_norm else k_norm.insertAxes(.k, .{.b});
    const v_unpadded = if (had_batch) v_norm else v_norm.insertAxes(.k, .{.b});

    const full_sprior_unpadded = k_unpadded.dim(.k);
    const full_sprior = alignDecodeSprior(full_sprior_unpadded);
    const padded_shape = zml.Shape.init(.{
        .b = k_unpadded.dim(.b),
        .k = full_sprior - full_sprior_unpadded,
        .h = k_unpadded.dim(.h),
        .hd = k_unpadded.dim(.hd),
    }, k_unpadded.dtype());
    const k_batched = if (full_sprior == full_sprior_unpadded)
        k_unpadded
    else
        zml.Tensor.concatenate(&.{
            k_unpadded,
            zml.Tensor.constant(k_unpadded.dtype().zero()).broad(padded_shape),
        }, .k);
    const v_batched = if (full_sprior == full_sprior_unpadded)
        v_unpadded
    else
        zml.Tensor.concatenate(&.{
            v_unpadded,
            zml.Tensor.constant(v_unpadded.dtype().zero()).broad(padded_shape),
        }, .k);

    const batch_size = q_batched.dim(.b);
    const num_heads = q_batched.dim(.h);
    const num_kv_heads = k_batched.dim(.h);
    const heads_per_kv = @divExact(num_heads, num_kv_heads);
    const head_dim = q_batched.dim(.hd);
    const effective_batch = batch_size * num_kv_heads;
    const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(head_dim))));
    const kernel_dtype = q_batched.dtype();

    const q_grouped = q_batched.convert(kernel_dtype).splitAxis(.h, .{ .kvh = num_kv_heads, .hq = .auto });
    const q_tkg = q_grouped.scale(scale)
        .transpose(.{ .hd, .b, .kvh, .hq, .q })
        .merge(.{ .beff = .{ .b, .kvh } })
        .merge(.{ .tot = .{ .beff, .hq, .q } })
        .rename(.{ .hd = .d });

    const active_k = k_batched.convert(kernel_dtype).dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = token_index_scalar, .len = 1 } });
    const active_v = v_batched.convert(kernel_dtype).dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = token_index_scalar, .len = 1 } });

    const k_active_tkg = active_k.transpose(.{ .hd, .b, .h, .k })
        .merge(.{ .tot = .{ .b, .h, .k } })
        .rename(.{ .hd = .d });
    const v_active_tkg = active_v.transpose(.{ .b, .h, .k, .hd })
        .merge(.{ .beff = .{ .b, .h } })
        .rename(.{ .beff = .b, .hd = .d })
        .insertAxes(.k, .{.one});

    const k_prior_tkg = k_batched.convert(kernel_dtype).transpose(.{ .b, .h, .hd, .k })
        .merge(.{ .beff = .{ .b, .h } })
        .rename(.{ .beff = .b, .hd = .d })
        .insertAxes(.d, .{.one});
    const v_prior_tkg = v_batched.convert(kernel_dtype).transpose(.{ .b, .h, .k, .hd })
        .merge(.{ .beff = .{ .b, .h } })
        .rename(.{ .beff = .b, .hd = .d })
        .insertAxes(.k, .{.one});

    const mask_shape_dtype = zml.Shape.init(.{ .k = full_sprior, .b = effective_batch, .hq = heads_per_kv, .q = 1 }, token_index_scalar.dtype());
    const mask_shape_u8 = zml.Shape.init(.{ .k = full_sprior, .b = effective_batch, .hq = heads_per_kv, .q = 1 }, .u8);
    const prior_positions = zml.Tensor.iota(zml.Shape.init(.{ .k = full_sprior }, .i32), .k).convert(token_index_scalar.dtype());
    const prior_mask = prior_positions
        .insertAxes(.last, .{ .b, .hq, .q })
        .broad(mask_shape_dtype)
        .cmp(.LT, token_index_scalar);
    const mask = prior_mask.select(
        zml.Tensor.constant(.{ .u8 = 1 }).broad(mask_shape_u8),
        zml.Tensor.constant(.{ .u8 = 0 }).broad(mask_shape_u8),
    );

    var out = zml.ops.neuronNki(
        .{ q_tkg, k_active_tkg, v_active_tkg, k_prior_tkg, v_prior_tkg, mask },
        .{zml.Shape.init(.{ .b = effective_batch, .hq = heads_per_kv, .d = head_dim, .q = 1 }, kernel_dtype)},
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

    return if (had_batch) restoreBatchTag(out, batch_tag) else out.squeeze(.b);
}

pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    if (!canUseAttentionTkg(q, k, token_index)) {
        log.warn("falling back to vanilla attention on neuron q={} k={} h={} kv_h={} hd={}", .{
            q.dim(.q),
            k.dim(.k),
            q.dim(.h),
            k.dim(.h),
            q.dim(.hd),
        });
        return zml.nn.sdpa(q, k, v, .{
            .attn_mask = zml.nn.causalAttnMask(.{ .q = k.dim(.k), .k = k.dim(.k) }, q.dtype(), null)
                .gatherSlices(zml.Shape.init(.{ .q = q.dim(.q) }, q.dtype()), normalizeTokenIndex(token_index).reshape(.{ .coord = 1 }), .{}),
            .allow_cudnn = true,
        });
    }

    const num_heads = q.dim(.h);
    const num_kv_heads = k.dim(.h);
    const heads_per_kv = @divExact(num_heads, num_kv_heads);
    const max_q_heads_per_chunk: i64 = 8;
    const max_kv_heads_per_chunk: i64 = 2;

    const max_kv_from_q_budget = @max(@as(i64, 1), @divFloor(max_q_heads_per_chunk, heads_per_kv));
    const kv_heads_per_chunk = blk: {
        var candidate = @min(num_kv_heads, @min(max_kv_heads_per_chunk, max_kv_from_q_budget));
        while (candidate > 1 and @mod(num_kv_heads, candidate) != 0) : (candidate -= 1) {}
        break :blk candidate;
    };
    const q_heads_per_chunk = heads_per_kv * kv_heads_per_chunk;

    log.warn("neuron attention decode config q={} k={} h={} kv_h={} hd={} heads_per_kv={} q_chunk={} kv_chunk={} chunks={}", .{
        q.dim(.q),
        k.dim(.k),
        num_heads,
        num_kv_heads,
        q.dim(.hd),
        heads_per_kv,
        q_heads_per_chunk,
        kv_heads_per_chunk,
        @divExact(num_kv_heads, kv_heads_per_chunk),
    });

    if (num_heads <= max_q_heads_per_chunk and num_kv_heads <= max_kv_heads_per_chunk) {
        return decodeAttentionTkgUnchunked(q, k, v, token_index);
    }

    var outputs: [16]zml.Tensor = undefined;
    const num_chunks: usize = @intCast(@divExact(num_kv_heads, kv_heads_per_chunk));
    std.debug.assert(num_chunks <= outputs.len);

    var chunk_idx: usize = 0;
    while (chunk_idx < num_chunks) : (chunk_idx += 1) {
        const kv_chunk_start = @as(i64, @intCast(chunk_idx)) * kv_heads_per_chunk;
        const q_chunk_start = @as(i64, @intCast(chunk_idx)) * q_heads_per_chunk;
        log.warn("neuron attention decode chunk idx={} q_start={} q_end={} kv_start={} kv_end={}", .{
            chunk_idx,
            q_chunk_start,
            q_chunk_start + q_heads_per_chunk,
            kv_chunk_start,
            kv_chunk_start + kv_heads_per_chunk,
        });
        outputs[chunk_idx] = decodeAttentionTkgUnchunked(
            q.slice1d(.h, .{ .start = q_chunk_start, .end = q_chunk_start + q_heads_per_chunk }),
            k.slice1d(.h, .{ .start = kv_chunk_start, .end = kv_chunk_start + kv_heads_per_chunk }),
            v.slice1d(.h, .{ .start = kv_chunk_start, .end = kv_chunk_start + kv_heads_per_chunk }),
            token_index,
        );
    }

    return zml.Tensor.concatenate(outputs[0..num_chunks], .h);
}
