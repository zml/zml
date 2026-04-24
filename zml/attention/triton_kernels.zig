//! TTIR generation for unified paged attention — direct port of the
//! `@triton.jit` kernels in `triton_kernels/unified_attention.py` (adapted
//! from vLLM). Each top-level function here mirrors a Python kernel 1:1:
//! same name (snake_case → camelCase), same parameter intent, same body
//! flow. Helper functions (`fastExp`, `cdivFn`, `applySoftcap`,
//! `findSeqIdx`) take explicit `*Kernel` + `Value` arguments — no
//! anytype/context plumbing.
//!
//! The generated IR is handed to `zml.ops.triton(...)` by
//! `zml/attention/triton.zig`.

const std = @import("std");

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");
const ttir = dialects.ttir;

const zml = @import("../zml.zig");
const tri = @import("zml/triton");
const Kernel = tri.Kernel;
const Value = tri.Value;
const DType = tri.DType;

const log = std.log.scoped(.@"zml/attention/triton");

// ============================================================================
// Constants
// ============================================================================

pub const FP8_E4M3_MIN: f32 = -448.0;
pub const FP8_E4M3_MAX: f32 = 448.0;

/// `log_2(e)` — pre-multiply with this to turn `exp(x)` into `exp2(x * RCP_LN2)`.
const RCP_LN2: f32 = 1.4426950408889634;

// ============================================================================
// Per-kernel configs — the `tl.constexpr` surface of each Python kernel.
// ============================================================================

/// Compile-time surface of `kernel_unified_attention_2d` and its `_ptr`
/// wrapper. Matches the `tl.constexpr` params of the Python kernel.
pub const Config2D = struct {
    q_dtype: DType,
    kv_dtype: DType,
    o_dtype: DType,
    scale_dtype: DType = .f32,

    num_query_heads: i64,
    num_queries_per_kv: i64,

    block_size: i64,
    tile_size: i64,
    head_size: i64,
    head_size_padded: i64,

    use_alibi_slopes: bool,
    use_qq_bias: bool,
    use_softcap: bool,
    use_sinks: bool,
    sliding_window: i64,

    stride_k_cache_3: i64 = 1,
    stride_v_cache_3: i64 = 1,

    block_q: i64,
    block_m: i64,
    use_fp8: bool,
    fp8_min: f32 = FP8_E4M3_MIN,
    fp8_max: f32 = FP8_E4M3_MAX,
    all_decode: bool = false,
};

/// Compile-time surface of `kernel_unified_attention_3d` / `_ptr`.
pub const Config3D = struct {
    q_dtype: DType,
    kv_dtype: DType,
    scale_dtype: DType = .f32,

    num_query_heads: i64,
    num_queries_per_kv: i64,

    block_size: i64,
    tile_size: i64,
    head_size: i64,
    head_size_padded: i64,

    use_alibi_slopes: bool,
    use_qq_bias: bool,
    use_softcap: bool,
    use_sinks: bool,
    sliding_window: i64,

    stride_k_cache_3: i64 = 1,
    stride_v_cache_3: i64 = 1,

    block_q: i64,
    block_m: i64,
    num_segments_per_seq: i64,
    all_decode: bool = false,
};

/// Compile-time surface of `reduce_segments` / `_ptr`.
pub const ConfigReduce = struct {
    o_dtype: DType,
    scale_dtype: DType = .f32,

    num_query_heads: i64,
    tile_size: i64,
    head_size: i64,
    head_size_padded: i64,

    block_q: i64,
    num_segments_per_seq: i64,

    use_fp8: bool,
    fp8_min: f32 = FP8_E4M3_MIN,
    fp8_max: f32 = FP8_E4M3_MAX,
};

// ============================================================================
// Helpers
// ============================================================================

fn ctx() *mlir.Context {
    return zml.module.CompilationContext.current().mlir_ctx;
}

fn isFp8(dt: DType) bool {
    return dt == .f8e4m3fn or dt == .f8e5m2;
}

/// Python `fast_exp(x)` — `exp2(x * RCP_LN2)`.
fn fastExp(k: *Kernel, x: Value) Value {
    return k.exp2(x.mul(RCP_LN2));
}

/// Python `cdiv_fn(x, y)` — ceiling integer divide for positive operands.
fn cdivFn(x: Value, y: anytype) Value {
    return x.cdiv(y);
}

/// Python `apply_softcap(S, x)` — tanh-style softcap using exp2.
fn applySoftcap(k: *Kernel, s_val: Value, x: Value) Value {
    const sdiv = s_val.div(x);
    const p1 = k.exp2(sdiv);
    const p2 = k.exp2(k.negf(sdiv));
    return x.mul(p1.sub(p2)).div(p1.add(p2));
}

/// Python `find_seq_idx(query_start_len_ptr, target_idx, num_seqs, BLOCK_Q,
/// use_q_block_mode)` — binary search into the CSR-style offset table.
fn findSeqIdx(
    k: *Kernel,
    query_start_len_ptr: Value,
    target_idx: Value,
    num_seqs: Value,
    block_q: i64,
    use_q_block_mode: bool,
) Value {
    const left_init = k.liftAs(0, .i32);

    var w = k.openWhile(.{ left_init, num_seqs }, .{ k.scalarTy(.i32), k.scalarTy(.i32) });
    {
        const left = w.before_carried[0];
        const right = w.before_carried[1];
        w.yieldBefore(left.lt(right), .{ left, right });
    }
    {
        const left = w.after_carried[0];
        const right = w.after_carried[1];
        const mid = left.add(right).div(2);
        const val = k.load(query_start_len_ptr.addPtr(mid));
        const mid_val = if (use_q_block_mode)
            val.div(@as(i32, @intCast(block_q))).add(mid)
        else
            val;

        var i = k.openIfElse(mid_val.le(target_idx), .{ k.scalarTy(.i32), k.scalarTy(.i32) });
        {
            i.yieldThen(.{ mid.add(1), right });
        }
        {
            i.yieldElse(.{ left, mid });
        }
        w.yieldAfter(.{ i.results[0], i.results[1] });
    }

    return w.results[0].sub(1);
}

/// Convert K/V loaded values to Q's dtype with scale applied when storage is
/// fp8 and Q isn't. Mirrors the Python `if K_load.dtype.is_fp8(): ...` gate.
fn dequantKv(loaded: Value, scale: Value, kv_is_fp8: bool, q_dtype: DType) Value {
    if (!kv_is_fp8 or isFp8(q_dtype)) return loaded;
    return loaded.to(.f32).mul(scale).to(q_dtype);
}

// ============================================================================
// kernel_unified_attention_2d — inner body (no pointer-unwrap prelude).
// ============================================================================

/// Emit the body of `kernel_unified_attention_2d`. Scalars (`scale`,
/// `*_scale`, `softcap`, strides, `num_seqs`) are passed as Values so this
/// helper can be called from both the scalar- and pointer-arg variants.
fn kernelUnifiedAttention2d(
    k: *Kernel,
    output_ptr: Value,
    query_ptr: Value,
    key_cache_ptr: Value,
    value_cache_ptr: Value,
    sink_ptr: Value,
    block_tables_ptr: Value,
    seq_lens_ptr: Value,
    alibi_slopes_ptr: Value,
    qq_bias_ptr: Value,
    scale: Value,
    k_scale: Value,
    v_scale: Value,
    out_scale: Value,
    softcap: Value,
    block_table_stride: Value,
    query_stride_0: Value,
    query_stride_1: Value,
    output_stride_0: Value,
    output_stride_1: Value,
    qq_bias_stride_0: Value,
    stride_k_cache_0: Value,
    stride_k_cache_1: Value,
    stride_k_cache_2: Value,
    stride_v_cache_0: Value,
    stride_v_cache_1: Value,
    stride_v_cache_2: Value,
    query_start_len_ptr: Value,
    num_seqs: Value,
    config: Config2D,
) void {
    const BLOCK_M: i64 = config.block_m;
    const BLOCK_Q: i64 = config.block_q;
    const BLOCK_SIZE: i64 = config.block_size;
    const TILE_SIZE: i64 = config.tile_size;
    const HEAD_SIZE: i64 = config.head_size;
    const HEAD_SIZE_PADDED: i64 = config.head_size_padded;
    const NUM_QUERIES_PER_KV: i64 = config.num_queries_per_kv;
    const NUM_QUERY_HEADS: i64 = config.num_query_heads;
    const SLIDING_WINDOW: i64 = config.sliding_window;

    const kv_head_idx = k.programId(.x);
    const q_block_global_idx = k.programId(.y);

    const qk_scale = scale.mul(RCP_LN2);

    const seq_idx = findSeqIdx(k, query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, true);

    const q_start_raw = k.load(query_start_len_ptr.addPtr(seq_idx));
    const q_block_start_idx = q_start_raw.div(@as(i32, @intCast(BLOCK_Q))).add(seq_idx);

    const q_block_local_idx = q_block_global_idx.sub(q_block_start_idx);

    const cur_batch_in_all_start_index = q_start_raw;
    const cur_batch_in_all_stop_index = k.load(query_start_len_ptr.addPtr(seq_idx.add(1)));
    const cur_batch_query_len = cur_batch_in_all_stop_index.sub(cur_batch_in_all_start_index);

    // Python: if q_block_local_idx * BLOCK_Q >= cur_batch_query_len: return
    const out_of_range = q_block_local_idx.mul(@as(i32, @intCast(BLOCK_Q))).ge(cur_batch_query_len);
    k.returnIf(out_of_range, .{});

    const offs_m = k.arange(0, BLOCK_M, .i32);
    const offs_d = k.arange(0, HEAD_SIZE_PADDED, .i32);
    const offs_t = k.arange(0, TILE_SIZE, .i32);

    // query_pos = q_block_local_idx * BLOCK_Q + offs_m // NUM_QUERIES_PER_KV
    const q_local_bq = q_block_local_idx.mul(@as(i32, @intCast(BLOCK_Q)));
    const query_pos = q_local_bq.add(offs_m.div(@as(i32, @intCast(NUM_QUERIES_PER_KV))));

    const query_offset_0 = cur_batch_in_all_start_index.add(query_pos).to(.i64);
    const query_offset_1 = kv_head_idx.mul(@as(i32, @intCast(NUM_QUERIES_PER_KV)))
        .add(offs_m.rem(@as(i32, @intCast(NUM_QUERIES_PER_KV)))).to(.i64);

    // query_offset = qo_0[:, None]*qstride_0 + qo_1[:, None]*qstride_1 + offs_d[None, :]
    const qo0_2d = query_offset_0.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED).mul(query_stride_0);
    const qo1_2d = query_offset_1.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED).mul(query_stride_1);
    const offs_d_2d = offs_d.to(.i64).broadcast2d(0, BLOCK_M, HEAD_SIZE_PADDED);
    const query_offset = qo0_2d.add(qo1_2d).add(offs_d_2d);

    const dim_mask: Value = if (HEAD_SIZE_PADDED != HEAD_SIZE)
        offs_d.lt(@as(i32, @intCast(HEAD_SIZE)))
    else
        k.full(&.{HEAD_SIZE_PADDED}, 1, .i1);
    const query_mask_0 = query_pos.lt(cur_batch_query_len);
    const query_mask_1 = query_offset_1.lt(NUM_QUERY_HEADS);

    // Q : (BLOCK_M, HEAD_SIZE_PADDED)
    const q_mask_ab = k.mask2d(query_mask_0, dim_mask, BLOCK_M, HEAD_SIZE_PADDED);
    const q_mask_c = query_mask_1.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED);
    const q_mask = q_mask_ab.bitAnd(q_mask_c);
    const q_cache_mod: ttir.CacheModifier =
        if (config.all_decode or BLOCK_M >= NUM_QUERY_HEADS) .cg else .none;
    const Q = k.loadOpts(query_ptr.addPtr(query_offset), .{
        .mask = q_mask,
        .other = k.zeros(&.{ BLOCK_M, HEAD_SIZE_PADDED }, config.q_dtype),
        .cache_modifier = q_cache_mod,
    });

    const block_table_offset = seq_idx.to(.i64).mul(block_table_stride);

    // M init — sinks or -inf.
    const m_init: Value = if (config.use_sinks) mb: {
        const loaded = k.loadOpts(sink_ptr.addPtr(query_offset_1), .{
            .mask = query_mask_1,
            .other = k.full(&.{BLOCK_M}, -std.math.inf(f32), config.scale_dtype),
        });
        break :mb loaded.to(.f32).mul(RCP_LN2);
    } else k.full(&.{BLOCK_M}, -std.math.inf(f32), .f32);

    const l_init = k.full(&.{BLOCK_M}, 1.0, .f32);
    const acc_init = k.zeros(&.{ BLOCK_M, HEAD_SIZE_PADDED }, .f32);

    const seq_len = k.load(seq_lens_ptr.addPtr(seq_idx));
    const context_len = seq_len.sub(cur_batch_query_len);

    const alibi_slope: Value = if (config.use_alibi_slopes)
        k.loadOpts(alibi_slopes_ptr.addPtr(query_offset_1), .{
            .mask = query_mask_1,
            .other = k.full(&.{BLOCK_M}, 0.0, config.scale_dtype),
        })
    else
        k.zeros(&.{BLOCK_M}, config.scale_dtype);

    // In the `use_qq_bias = true` branch, the tensor offset auto-splats the
    // scalar ptr via addPtr. In the else branch there is no offset, so we
    // need an explicit splatTo to give broadcast2d a tensor to work on.
    const qq_bias_row_ptrs: Value = if (config.use_qq_bias)
        qq_bias_ptr.addPtr(query_pos.to(.i64).mul(qq_bias_stride_0))
    else
        qq_bias_ptr.splatTo(&.{BLOCK_M});

    // max_seq_prefix_len = context_len + q_block_local_idx*BLOCK_Q + (BLOCK_M-1)/NQ_PER_KV + 1
    const pad_term: i32 = @intCast(@divTrunc(BLOCK_M - 1, NUM_QUERIES_PER_KV) + 1);
    const max_prefix_raw = context_len.add(q_local_bq).add(pad_term);
    const max_seq_prefix_len = max_prefix_raw.minimum(seq_len);

    const num_tiles = cdivFn(max_seq_prefix_len, @as(i32, @intCast(TILE_SIZE)));

    // Sliding-window tile pruning — comptime branch.
    var tile_start: Value = k.liftAs(0, .i32);
    var tile_end: Value = num_tiles;
    if (SLIDING_WINDOW > 0) {
        const qpos_lo = q_local_bq;
        const qpos_hi_raw = qpos_lo.add(pad_term - 1);
        const qpos_hi = qpos_hi_raw.minimum(cur_batch_query_len.sub(1));
        const first_allowed_key = context_len.add(qpos_lo).sub(@as(i32, @intCast(SLIDING_WINDOW - 1)));
        const last_allowed_key = context_len.add(qpos_hi);
        tile_start = first_allowed_key.div(@as(i32, @intCast(TILE_SIZE))).maximum(0);
        tile_end = last_allowed_key.div(@as(i32, @intCast(TILE_SIZE))).add(1).minimum(num_tiles);
    }

    const kv_cache_mod: ttir.CacheModifier =
        if (config.all_decode) .cg else .none;

    // Iterate through tiles (loop carries M, L, acc).
    var loop = k.openFor(tile_start, tile_end, 1, .{ m_init, l_init, acc_init });
    {
        const j = loop.iv;
        const M = loop.carried[0];
        const L = loop.carried[1];
        const acc = loop.carried[2];

        const seq_offset = j.mul(@as(i32, @intCast(TILE_SIZE))).add(offs_t);

        const tile_mask: Value = if (TILE_SIZE == BLOCK_SIZE)
            k.full(&.{TILE_SIZE}, 1, .i1)
        else
            seq_offset.lt(max_seq_prefix_len);

        const physical_block_idx = k.load(
            block_tables_ptr.addPtr(block_table_offset.add(seq_offset.to(.i64).div(BLOCK_SIZE))),
        ).to(.i64);

        const seq_in_block = seq_offset.to(.i64).rem(BLOCK_SIZE);

        // v_offset : (TILE_SIZE, HEAD_SIZE_PADDED)
        const pb_v = physical_block_idx.broadcast2d(1, TILE_SIZE, HEAD_SIZE_PADDED).mul(stride_v_cache_0);
        const head_v = kv_head_idx.to(.i64).mul(stride_v_cache_2);
        const dim_v = offs_d.to(.i64).broadcast2d(0, TILE_SIZE, HEAD_SIZE_PADDED)
            .mul(config.stride_v_cache_3);
        const blk_v = seq_in_block.broadcast2d(1, TILE_SIZE, HEAD_SIZE_PADDED).mul(stride_v_cache_1);
        const v_offset = pb_v.add(head_v).add(dim_v).add(blk_v);

        // k_offset : (HEAD_SIZE_PADDED, TILE_SIZE)
        const pb_k = physical_block_idx.broadcast2d(0, HEAD_SIZE_PADDED, TILE_SIZE).mul(stride_k_cache_0);
        const head_k = kv_head_idx.to(.i64).mul(stride_k_cache_2);
        const dim_k = offs_d.to(.i64).broadcast2d(1, HEAD_SIZE_PADDED, TILE_SIZE)
            .mul(config.stride_k_cache_3);
        const blk_k = seq_in_block.broadcast2d(0, HEAD_SIZE_PADDED, TILE_SIZE).mul(stride_k_cache_1);
        const k_offset = pb_k.add(head_k).add(dim_k).add(blk_k);

        // K : (HEAD_SIZE_PADDED, TILE_SIZE)
        const k_mask = k.mask2d(dim_mask, tile_mask, HEAD_SIZE_PADDED, TILE_SIZE);
        const K_load = k.loadOpts(key_cache_ptr.addPtr(k_offset), .{
            .mask = k_mask,
            .other = k.zeros(&.{ HEAD_SIZE_PADDED, TILE_SIZE }, config.kv_dtype),
            .cache_modifier = kv_cache_mod,
        });
        const K = dequantKv(K_load, k_scale, isFp8(config.kv_dtype), config.q_dtype);

        // V : (TILE_SIZE, HEAD_SIZE_PADDED)
        const v_mask = k.mask2d(tile_mask, dim_mask, TILE_SIZE, HEAD_SIZE_PADDED);
        const V_load = k.loadOpts(value_cache_ptr.addPtr(v_offset), .{
            .mask = v_mask,
            .other = k.zeros(&.{ TILE_SIZE, HEAD_SIZE_PADDED }, config.kv_dtype),
            .cache_modifier = kv_cache_mod,
        });
        const V = dequantKv(V_load, v_scale, isFp8(config.kv_dtype), config.q_dtype);

        // S : (BLOCK_M, TILE_SIZE)
        const acc_zero = k.zeros(&.{ BLOCK_M, TILE_SIZE }, .f32);
        const qk = k.dot(Q, K, acc_zero);
        var S = qk_scale.mul(qk);

        if (config.use_softcap) {
            S = applySoftcap(k, S, softcap).mul(RCP_LN2);
        }

        // seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
        const ql_2d_i32 = query_pos.broadcast2d(1, BLOCK_M, TILE_SIZE);
        const so_2d = seq_offset.broadcast2d(0, BLOCK_M, TILE_SIZE);
        const rhs = ql_2d_i32.add(context_len).add(1);
        const seq_mask = so_2d.lt(rhs);

        // S = tl.where(qmask_1 & qmask_0 & seq_mask, S, -inf)
        const qm0_2d = query_mask_0.broadcast2d(1, BLOCK_M, TILE_SIZE);
        const qm1_2d = query_mask_1.broadcast2d(1, BLOCK_M, TILE_SIZE);
        const keep_mask = qm1_2d.bitAnd(qm0_2d).bitAnd(seq_mask);
        S = k.where(keep_mask, S, k.full(&.{ BLOCK_M, TILE_SIZE }, -std.math.inf(f32), .f32));

        if (SLIDING_WINDOW > 0) {
            const diff = ql_2d_i32.add(context_len).sub(so_2d);
            const in_win = diff.lt(@as(i32, @intCast(SLIDING_WINDOW)));
            S = k.where(in_win, S, k.full(&.{ BLOCK_M, TILE_SIZE }, -std.math.inf(f32), .f32));
        }

        if (config.use_alibi_slopes) {
            const alibi_2d = alibi_slope.broadcast2d(1, BLOCK_M, TILE_SIZE);
            const pos_diff = seq_offset.sub(context_len).to(config.scale_dtype).broadcast2d(0, BLOCK_M, TILE_SIZE);
            S = S.add(alibi_2d.mul(pos_diff).to(.f32).mul(RCP_LN2));
        }

        if (config.use_qq_bias) {
            const key_rel_pos = seq_offset.sub(context_len);
            const is_query_key = key_rel_pos.ge(0)
                .bitAnd(key_rel_pos.to(.i64).lt(qq_bias_stride_0));
            const qq_ptrs = qq_bias_row_ptrs.broadcast2d(1, BLOCK_M, TILE_SIZE)
                .addPtr(key_rel_pos.to(.i64).broadcast2d(0, BLOCK_M, TILE_SIZE));
            const qq_bias = k.loadOpts(qq_ptrs, .{
                .mask = is_query_key.broadcast2d(0, BLOCK_M, TILE_SIZE),
                .other = k.zeros(&.{ BLOCK_M, TILE_SIZE }, config.scale_dtype),
            });
            S = S.add(qq_bias.to(.f32).mul(RCP_LN2));
        }

        // m_j = max(M, max(S, axis=1))
        var m_j = M.maximum(k.maxOpts(S, .{ .axis = 1 }));
        m_j = k.where(m_j.gt(-std.math.inf(f32)), m_j, k.full(&.{BLOCK_M}, 0.0, .f32));

        // P = exp2(S - m_j[:, None]); l_j = sum(P, axis=1); alpha = exp2(M - m_j)
        const mj_2d = m_j.broadcast2d(1, BLOCK_M, TILE_SIZE);
        const P = k.exp2(S.sub(mj_2d));
        const l_j = k.sumOpts(P, .{ .axis = 1 });
        const alpha = k.exp2(M.sub(m_j));

        const alpha_2d = alpha.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED);
        const acc_scaled = acc.mul(alpha_2d);
        const new_L = L.mul(alpha).add(l_j);
        const new_M = m_j;

        const P_cast = P.to(config.q_dtype);
        const new_acc = k.dot(P_cast, V, acc_scaled);

        loop.yield(.{ new_M, new_L, new_acc });
    }
    const L_final = loop.results[1];
    var acc_final = loop.results[2];

    // Epilogue: acc /= L (via 1/L for Newton-Raphson stability).
    const one_over_L = k.full(&.{BLOCK_M}, 1.0, .f32).div(L_final);
    acc_final = acc_final.mul(one_over_L.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED));

    if (config.use_fp8) {
        acc_final = acc_final.mul(out_scale);
        acc_final = k.clampf(
            acc_final,
            k.full(&.{ BLOCK_M, HEAD_SIZE_PADDED }, config.fp8_min, .f32),
            k.full(&.{ BLOCK_M, HEAD_SIZE_PADDED }, config.fp8_max, .f32),
        );
    }

    const oo0_2d = query_offset_0.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED).mul(output_stride_0);
    const oo1_2d = query_offset_1.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED).mul(output_stride_1);
    const output_offset = oo0_2d.add(oo1_2d).add(offs_d_2d);

    const store_mask = k.mask2d(query_mask_0, dim_mask, BLOCK_M, HEAD_SIZE_PADDED)
        .bitAnd(query_mask_1.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED));
    k.storeOpts(
        output_ptr.addPtr(output_offset),
        acc_final.to(config.o_dtype),
        .{ .mask = store_mask },
    );
}

// ============================================================================
// kernel_unified_attention_3d — inner body.
// ============================================================================

fn kernelUnifiedAttention3d(
    k: *Kernel,
    segm_output_ptr: Value,
    segm_max_ptr: Value,
    segm_expsum_ptr: Value,
    query_ptr: Value,
    key_cache_ptr: Value,
    value_cache_ptr: Value,
    sink_ptr: Value,
    block_tables_ptr: Value,
    seq_lens_ptr: Value,
    alibi_slopes_ptr: Value,
    qq_bias_ptr: Value,
    scale: Value,
    k_scale: Value,
    v_scale: Value,
    softcap: Value,
    block_table_stride: Value,
    query_stride_0: Value,
    query_stride_1: Value,
    qq_bias_stride_0: Value,
    stride_k_cache_0: Value,
    stride_k_cache_1: Value,
    stride_k_cache_2: Value,
    stride_v_cache_0: Value,
    stride_v_cache_1: Value,
    stride_v_cache_2: Value,
    query_start_len_ptr: Value,
    num_seqs: Value,
    config: Config3D,
) void {
    const BLOCK_M: i64 = config.block_m;
    const BLOCK_Q: i64 = config.block_q;
    const BLOCK_SIZE: i64 = config.block_size;
    const TILE_SIZE: i64 = config.tile_size;
    const HEAD_SIZE: i64 = config.head_size;
    const HEAD_SIZE_PADDED: i64 = config.head_size_padded;
    const NUM_QUERIES_PER_KV: i64 = config.num_queries_per_kv;
    const NUM_QUERY_HEADS: i64 = config.num_query_heads;
    const NUM_SEGMENTS_PER_SEQ: i64 = config.num_segments_per_seq;
    const SLIDING_WINDOW: i64 = config.sliding_window;

    const q_block_global_idx = k.programId(.x);
    const kv_head_idx = k.programId(.y);
    const segm_idx = k.programId(.z);

    const qk_scale = scale.mul(RCP_LN2);

    const seq_idx = findSeqIdx(k, query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, true);

    const q_start_raw = k.load(query_start_len_ptr.addPtr(seq_idx));
    const q_block_start_idx = q_start_raw.div(@as(i32, @intCast(BLOCK_Q))).add(seq_idx);
    const q_block_local_idx = q_block_global_idx.sub(q_block_start_idx);

    const cur_batch_in_all_start_index = q_start_raw;
    const cur_batch_in_all_stop_index = k.load(query_start_len_ptr.addPtr(seq_idx.add(1)));
    const cur_batch_query_len = cur_batch_in_all_stop_index.sub(cur_batch_in_all_start_index);

    const q_out_of_range = q_block_local_idx.mul(@as(i32, @intCast(BLOCK_Q))).ge(cur_batch_query_len);
    k.returnIf(q_out_of_range, .{});

    const seq_len = k.load(seq_lens_ptr.addPtr(seq_idx));
    const tiles_per_segment = cdivFn(seq_len, @as(i32, @intCast(NUM_SEGMENTS_PER_SEQ * TILE_SIZE)));

    const segm_lo = segm_idx.mul(tiles_per_segment).mul(@as(i32, @intCast(TILE_SIZE)));
    const segm_out_of_range = segm_lo.ge(seq_len);
    k.returnIf(segm_out_of_range, .{});

    const offs_m = k.arange(0, BLOCK_M, .i32);
    const offs_d = k.arange(0, HEAD_SIZE_PADDED, .i32);
    const offs_t = k.arange(0, TILE_SIZE, .i32);

    const q_local_bq = q_block_local_idx.mul(@as(i32, @intCast(BLOCK_Q)));
    const query_pos = q_local_bq.add(offs_m.div(@as(i32, @intCast(NUM_QUERIES_PER_KV))));

    const query_offset_0 = cur_batch_in_all_start_index.add(query_pos).to(.i64);
    const query_offset_1 = kv_head_idx.mul(@as(i32, @intCast(NUM_QUERIES_PER_KV)))
        .add(offs_m.rem(@as(i32, @intCast(NUM_QUERIES_PER_KV)))).to(.i64);

    const qo0_2d = query_offset_0.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED).mul(query_stride_0);
    const qo1_2d = query_offset_1.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED).mul(query_stride_1);
    const offs_d_2d = offs_d.to(.i64).broadcast2d(0, BLOCK_M, HEAD_SIZE_PADDED);
    const query_offset = qo0_2d.add(qo1_2d).add(offs_d_2d);

    const dim_mask: Value = if (HEAD_SIZE_PADDED != HEAD_SIZE)
        offs_d.lt(@as(i32, @intCast(HEAD_SIZE)))
    else
        k.full(&.{HEAD_SIZE_PADDED}, 1, .i1);
    const query_mask_0 = query_pos.lt(cur_batch_query_len);
    const query_mask_1 = query_offset_1.lt(NUM_QUERY_HEADS);

    const q_mask_ab = k.mask2d(query_mask_0, dim_mask, BLOCK_M, HEAD_SIZE_PADDED);
    const q_mask = q_mask_ab.bitAnd(query_mask_1.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED));
    const Q = k.loadOpts(query_ptr.addPtr(query_offset), .{
        .mask = q_mask,
        .other = k.zeros(&.{ BLOCK_M, HEAD_SIZE_PADDED }, config.q_dtype),
    });

    const block_table_offset = seq_idx.to(.i64).mul(block_table_stride);

    // M init: USE_SINKS + segm_idx==0 → loaded sinks; else -inf.
    const segm_is_zero = segm_idx.eq(0);
    const m_init: Value = if (config.use_sinks) mb: {
        var mi = k.openIfElse(segm_is_zero, .{k.tensorTy(&.{BLOCK_M}, .f32)});
        {
            const loaded = k.loadOpts(sink_ptr.addPtr(query_offset_1), .{
                .mask = query_mask_1,
                .other = k.full(&.{BLOCK_M}, -std.math.inf(f32), config.scale_dtype),
            });
            mi.yieldThen(.{loaded.to(.f32).mul(RCP_LN2)});
        }
        {
            mi.yieldElse(.{k.full(&.{BLOCK_M}, -std.math.inf(f32), .f32)});
        }
        break :mb mi.results[0];
    } else k.full(&.{BLOCK_M}, -std.math.inf(f32), .f32);

    const l_init = k.full(&.{BLOCK_M}, 1.0, .f32);
    const acc_init = k.zeros(&.{ BLOCK_M, HEAD_SIZE_PADDED }, .f32);

    const context_len = seq_len.sub(cur_batch_query_len);

    const alibi_slope: Value = if (config.use_alibi_slopes)
        k.loadOpts(alibi_slopes_ptr.addPtr(query_offset_1), .{
            .mask = query_mask_1,
            .other = k.full(&.{BLOCK_M}, 0.0, config.scale_dtype),
        })
    else
        k.zeros(&.{BLOCK_M}, config.scale_dtype);

    // See 2d-body note on qq_bias_row_ptrs: else branch needs an explicit splatTo
    // (no sibling offset to auto-broadcast against).
    const qq_bias_row_ptrs: Value = if (config.use_qq_bias)
        qq_bias_ptr.addPtr(query_pos.to(.i64).mul(qq_bias_stride_0))
    else
        qq_bias_ptr.splatTo(&.{BLOCK_M});

    const pad_term: i32 = @intCast(@divTrunc(BLOCK_M - 1, NUM_QUERIES_PER_KV) + 1);
    const max_prefix_raw = context_len.add(q_local_bq).add(pad_term);
    const max_seq_prefix_len = max_prefix_raw.minimum(seq_len);
    const num_tiles = cdivFn(max_seq_prefix_len, @as(i32, @intCast(TILE_SIZE)));

    const segm_tile_lo = segm_idx.mul(tiles_per_segment);
    const segm_tile_hi = segm_tile_lo.add(tiles_per_segment).minimum(num_tiles);

    const kv_cache_mod: ttir.CacheModifier =
        if (config.all_decode) .cg else .none;

    var loop = k.openFor(segm_tile_lo, segm_tile_hi, 1, .{ m_init, l_init, acc_init });
    {
        const j = loop.iv;
        const M = loop.carried[0];
        const L = loop.carried[1];
        const acc = loop.carried[2];

        const seq_offset = j.mul(@as(i32, @intCast(TILE_SIZE))).add(offs_t);
        const tile_mask: Value = if (TILE_SIZE == BLOCK_SIZE)
            k.full(&.{TILE_SIZE}, 1, .i1)
        else
            seq_offset.lt(max_seq_prefix_len);

        const physical_block_idx = k.load(
            block_tables_ptr.addPtr(block_table_offset.add(seq_offset.to(.i64).div(BLOCK_SIZE))),
        ).to(.i64);

        const seq_in_block = seq_offset.to(.i64).rem(BLOCK_SIZE);

        const pb_v = physical_block_idx.broadcast2d(1, TILE_SIZE, HEAD_SIZE_PADDED).mul(stride_v_cache_0);
        const head_v = kv_head_idx.to(.i64).mul(stride_v_cache_2);
        const dim_v = offs_d.to(.i64).broadcast2d(0, TILE_SIZE, HEAD_SIZE_PADDED)
            .mul(config.stride_v_cache_3);
        const blk_v = seq_in_block.broadcast2d(1, TILE_SIZE, HEAD_SIZE_PADDED).mul(stride_v_cache_1);
        const v_offset = pb_v.add(head_v).add(dim_v).add(blk_v);

        const pb_k = physical_block_idx.broadcast2d(0, HEAD_SIZE_PADDED, TILE_SIZE).mul(stride_k_cache_0);
        const head_k = kv_head_idx.to(.i64).mul(stride_k_cache_2);
        const dim_k = offs_d.to(.i64).broadcast2d(1, HEAD_SIZE_PADDED, TILE_SIZE)
            .mul(config.stride_k_cache_3);
        const blk_k = seq_in_block.broadcast2d(0, HEAD_SIZE_PADDED, TILE_SIZE).mul(stride_k_cache_1);
        const k_offset = pb_k.add(head_k).add(dim_k).add(blk_k);

        const k_mask = k.mask2d(dim_mask, tile_mask, HEAD_SIZE_PADDED, TILE_SIZE);
        const K_load = k.loadOpts(key_cache_ptr.addPtr(k_offset), .{
            .mask = k_mask,
            .other = k.zeros(&.{ HEAD_SIZE_PADDED, TILE_SIZE }, config.kv_dtype),
            .cache_modifier = kv_cache_mod,
        });
        const K = dequantKv(K_load, k_scale, isFp8(config.kv_dtype), config.q_dtype);

        const v_mask = k.mask2d(tile_mask, dim_mask, TILE_SIZE, HEAD_SIZE_PADDED);
        const V_load = k.loadOpts(value_cache_ptr.addPtr(v_offset), .{
            .mask = v_mask,
            .other = k.zeros(&.{ TILE_SIZE, HEAD_SIZE_PADDED }, config.kv_dtype),
            .cache_modifier = kv_cache_mod,
        });
        const V = dequantKv(V_load, v_scale, isFp8(config.kv_dtype), config.q_dtype);

        const acc_zero = k.zeros(&.{ BLOCK_M, TILE_SIZE }, .f32);
        const qk = k.dot(Q, K, acc_zero);
        var S = qk_scale.mul(qk);

        if (config.use_softcap) {
            S = applySoftcap(k, S, softcap).mul(RCP_LN2);
        }

        const ql_2d_i32 = query_pos.broadcast2d(1, BLOCK_M, TILE_SIZE);
        const so_2d = seq_offset.broadcast2d(0, BLOCK_M, TILE_SIZE);
        const rhs = ql_2d_i32.add(context_len).add(1);
        const seq_mask = so_2d.lt(rhs);

        const qm0_2d = query_mask_0.broadcast2d(1, BLOCK_M, TILE_SIZE);
        const qm1_2d = query_mask_1.broadcast2d(1, BLOCK_M, TILE_SIZE);
        const keep_mask = qm1_2d.bitAnd(qm0_2d).bitAnd(seq_mask);
        S = k.where(keep_mask, S, k.full(&.{ BLOCK_M, TILE_SIZE }, -std.math.inf(f32), .f32));

        if (SLIDING_WINDOW > 0) {
            const diff = ql_2d_i32.add(context_len).sub(so_2d);
            const in_win = diff.lt(@as(i32, @intCast(SLIDING_WINDOW)));
            S = k.where(in_win, S, k.full(&.{ BLOCK_M, TILE_SIZE }, -std.math.inf(f32), .f32));
        }

        if (config.use_alibi_slopes) {
            const alibi_2d = alibi_slope.broadcast2d(1, BLOCK_M, TILE_SIZE);
            const pos_diff = seq_offset.sub(context_len).to(config.scale_dtype).broadcast2d(0, BLOCK_M, TILE_SIZE);
            S = S.add(alibi_2d.mul(pos_diff).to(.f32).mul(RCP_LN2));
        }

        if (config.use_qq_bias) {
            const key_rel_pos = seq_offset.sub(context_len);
            const is_query_key = key_rel_pos.ge(0)
                .bitAnd(key_rel_pos.to(.i64).lt(qq_bias_stride_0));
            const qq_ptrs = qq_bias_row_ptrs.broadcast2d(1, BLOCK_M, TILE_SIZE)
                .addPtr(key_rel_pos.to(.i64).broadcast2d(0, BLOCK_M, TILE_SIZE));
            const qq_bias = k.loadOpts(qq_ptrs, .{
                .mask = is_query_key.broadcast2d(0, BLOCK_M, TILE_SIZE),
                .other = k.zeros(&.{ BLOCK_M, TILE_SIZE }, config.scale_dtype),
            });
            S = S.add(qq_bias.to(.f32).mul(RCP_LN2));
        }

        var m_j = M.maximum(k.maxOpts(S, .{ .axis = 1 }));
        m_j = k.where(m_j.gt(-std.math.inf(f32)), m_j, k.full(&.{BLOCK_M}, 0.0, .f32));

        const mj_2d = m_j.broadcast2d(1, BLOCK_M, TILE_SIZE);
        const P = k.exp2(S.sub(mj_2d));
        const l_j = k.sumOpts(P, .{ .axis = 1 });
        const alpha = k.exp2(M.sub(m_j));

        const alpha_2d = alpha.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED);
        const acc_scaled = acc.mul(alpha_2d);
        const new_L = L.mul(alpha).add(l_j);

        const P_cast = P.to(config.q_dtype);
        const new_acc = k.dot(P_cast, V, acc_scaled);

        loop.yield(.{ m_j, new_L, new_acc });
    }
    const M_final = loop.results[0];
    const L_final = loop.results[1];
    const acc_final = loop.results[2];

    // Store segm_output, segm_max, segm_expsum.
    const segm_out_head_stride: i64 = NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED;
    const segm_out_token_stride: i64 = NUM_QUERY_HEADS * segm_out_head_stride;

    const oo0_2d = query_offset_0.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED)
        .mul(segm_out_token_stride);
    const oo1_2d = query_offset_1.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED)
        .mul(segm_out_head_stride);
    const seg_term = segm_idx.to(.i64).mul(HEAD_SIZE_PADDED);
    const segm_output_offset = oo0_2d.add(oo1_2d).add(seg_term).add(offs_d_2d);

    const store_mask_2d = k.mask2d(query_mask_0, dim_mask, BLOCK_M, HEAD_SIZE_PADDED)
        .bitAnd(query_mask_1.broadcast2d(1, BLOCK_M, HEAD_SIZE_PADDED));
    k.storeOpts(
        segm_output_ptr.addPtr(segm_output_offset),
        acc_final,
        .{ .mask = store_mask_2d },
    );

    const segm_head_stride: i64 = NUM_SEGMENTS_PER_SEQ;
    const segm_token_stride: i64 = NUM_QUERY_HEADS * segm_head_stride;
    const segm_offset = query_offset_0.mul(segm_token_stride)
        .add(query_offset_1.mul(segm_head_stride))
        .add(segm_idx.to(.i64));
    const store_mask_1d = query_mask_0.bitAnd(query_mask_1);
    k.storeOpts(segm_max_ptr.addPtr(segm_offset), M_final, .{ .mask = store_mask_1d });
    k.storeOpts(segm_expsum_ptr.addPtr(segm_offset), L_final, .{ .mask = store_mask_1d });
}

// ============================================================================
// reduce_segments — inner body.
// ============================================================================

fn reduceSegments(
    k: *Kernel,
    output_ptr: Value,
    segm_output_ptr: Value,
    segm_max_ptr: Value,
    segm_expsum_ptr: Value,
    seq_lens_ptr: Value,
    num_seqs: Value,
    out_scale_inv: Value,
    output_stride_0: Value,
    output_stride_1: Value,
    block_table_stride: Value,
    query_start_len_ptr: Value,
    config: ConfigReduce,
) void {
    _ = block_table_stride;

    const TILE_SIZE: i64 = config.tile_size;
    const HEAD_SIZE: i64 = config.head_size;
    const HEAD_SIZE_PADDED: i64 = config.head_size_padded;
    const NUM_QUERY_HEADS: i64 = config.num_query_heads;
    const NUM_SEGMENTS_PER_SEQ: i64 = config.num_segments_per_seq;
    const BLOCK_Q: i64 = config.block_q;

    const query_token_idx = k.programId(.x);
    const query_head_idx = k.programId(.y);

    const seq_idx = findSeqIdx(k, query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, false);

    const seq_len = k.load(seq_lens_ptr.addPtr(seq_idx));
    const tiles_per_segment = cdivFn(seq_len, @as(i32, @intCast(NUM_SEGMENTS_PER_SEQ * TILE_SIZE)));
    const act_num_segments = cdivFn(seq_len, tiles_per_segment.mul(@as(i32, @intCast(TILE_SIZE))));

    const seg_range = k.arange(0, NUM_SEGMENTS_PER_SEQ, .i32);
    const segm_mask = seg_range.lt(act_num_segments);

    const offs_d = k.arange(0, HEAD_SIZE_PADDED, .i32);
    const dim_mask: Value = if (HEAD_SIZE_PADDED != HEAD_SIZE)
        offs_d.lt(@as(i32, @intCast(HEAD_SIZE)))
    else
        k.full(&.{HEAD_SIZE_PADDED}, 1, .i1);

    // segm_offset = query_token_idx*NQH*NSPS + query_head_idx*NSPS + arange(NSPS)
    const tok_stride: i64 = NUM_QUERY_HEADS * NUM_SEGMENTS_PER_SEQ;
    const head_stride: i64 = NUM_SEGMENTS_PER_SEQ;
    const segm_offset = query_token_idx.to(.i64).mul(tok_stride)
        .add(query_head_idx.to(.i64).mul(head_stride))
        .add(seg_range.to(.i64));

    const segm_max = k.loadOpts(segm_max_ptr.addPtr(segm_offset), .{
        .mask = segm_mask,
        .other = k.full(&.{NUM_SEGMENTS_PER_SEQ}, -std.math.inf(f32), .f32),
    });
    const overall_max = k.max(segm_max);

    var segm_expsum = k.loadOpts(segm_expsum_ptr.addPtr(segm_offset), .{
        .mask = segm_mask,
        .other = k.zeros(&.{NUM_SEGMENTS_PER_SEQ}, .f32),
    });
    segm_expsum = segm_expsum.mul(k.exp2(segm_max.sub(overall_max)));
    const overall_expsum = k.sum(segm_expsum);

    // segm_output_offset : (NSPS, HSP)
    const out_tok_stride: i64 = NUM_QUERY_HEADS * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED;
    const out_head_stride: i64 = NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED;
    const seg_off_2d = seg_range.to(.i64).broadcast2d(1, NUM_SEGMENTS_PER_SEQ, HEAD_SIZE_PADDED)
        .mul(HEAD_SIZE_PADDED);
    const dim_off_2d = offs_d.to(.i64).broadcast2d(0, NUM_SEGMENTS_PER_SEQ, HEAD_SIZE_PADDED);
    const base = query_token_idx.to(.i64).mul(out_tok_stride)
        .add(query_head_idx.to(.i64).mul(out_head_stride));
    const segm_output_offset = base.add(seg_off_2d).add(dim_off_2d);

    var segm_output = k.loadOpts(
        segm_output_ptr.addPtr(segm_output_offset),
        .{
            .mask = k.mask2d(segm_mask, dim_mask, NUM_SEGMENTS_PER_SEQ, HEAD_SIZE_PADDED),
            .other = k.zeros(&.{ NUM_SEGMENTS_PER_SEQ, HEAD_SIZE_PADDED }, .f32),
        },
    );
    const rescale = k.exp2(segm_max.sub(overall_max));
    segm_output = segm_output.mul(rescale.broadcast2d(1, NUM_SEGMENTS_PER_SEQ, HEAD_SIZE_PADDED));
    const acc_sum = k.sumOpts(segm_output, .{ .axis = 0 });

    var acc = k.where(
        overall_expsum.eq(0.0),
        k.full(&.{HEAD_SIZE_PADDED}, 0.0, .f32),
        acc_sum.div(overall_expsum),
    );

    if (config.use_fp8) {
        acc = acc.mul(out_scale_inv);
        acc = k.clampf(
            acc,
            k.full(&.{HEAD_SIZE_PADDED}, config.fp8_min, .f32),
            k.full(&.{HEAD_SIZE_PADDED}, config.fp8_max, .f32),
        );
    }

    const output_offset = query_token_idx.to(.i64).mul(output_stride_0)
        .add(query_head_idx.to(.i64).mul(output_stride_1))
        .add(offs_d.to(.i64));
    k.storeOpts(
        output_ptr.addPtr(output_offset),
        acc.to(config.o_dtype),
        .{ .mask = dim_mask },
    );
}

// ============================================================================
// Top-level `_ptr` wrappers — these are the kernels launched by the host.
// Each loads the runtime scalar args via `tl.load` and then emits the body.
// ============================================================================

/// Build the TTIR for `kernel_unified_attention_2d_ptr`.
pub fn kernelUnifiedAttention2dPtr(allocator: std.mem.Allocator, config: Config2D) ![:0]const u8 {
    var spec = try Kernel.build(allocator, ctx(), "kernel_unified_attention_2d_ptr", .{
        .query_ptr = .{ .ptr = config.q_dtype },
        .key_cache_ptr = .{ .ptr = config.kv_dtype },
        .value_cache_ptr = .{ .ptr = config.kv_dtype },
        .sink_ptr = .{ .ptr = config.scale_dtype },
        .block_tables_ptr = .{ .ptr = .i32 },
        .seq_lens_ptr = .{ .ptr = .i32 },
        .alibi_slopes_ptr = .{ .ptr = config.scale_dtype },
        .qq_bias_ptr = .{ .ptr = config.scale_dtype },
        .scale_ptr = .{ .ptr = .f32 },
        .k_scale_ptr = .{ .ptr = .f32 },
        .v_scale_ptr = .{ .ptr = .f32 },
        .out_scale_ptr = .{ .ptr = .f32 },
        .softcap_ptr = .{ .ptr = .f32 },
        .block_table_stride_ptr = .{ .ptr = .i64 },
        .query_stride_0_ptr = .{ .ptr = .i64 },
        .query_stride_1_ptr = .{ .ptr = .i64 },
        .output_stride_0_ptr = .{ .ptr = .i64 },
        .output_stride_1_ptr = .{ .ptr = .i64 },
        .qq_bias_stride_0_ptr = .{ .ptr = .i64 },
        .stride_k_cache_0_ptr = .{ .ptr = .i64 },
        .stride_k_cache_1_ptr = .{ .ptr = .i64 },
        .stride_k_cache_2_ptr = .{ .ptr = .i64 },
        .stride_v_cache_0_ptr = .{ .ptr = .i64 },
        .stride_v_cache_1_ptr = .{ .ptr = .i64 },
        .stride_v_cache_2_ptr = .{ .ptr = .i64 },
        .query_start_len_ptr = .{ .ptr = .i32 },
        .num_seqs_ptr = .{ .ptr = .i32 },
        .output_ptr = .{ .ptr = config.o_dtype },
    }, &.{});
    defer spec.deinit();
    const k = &spec.kernel;
    const a = spec.args;

    const scale = k.load(a.scale_ptr);
    const k_scale = k.load(a.k_scale_ptr);
    const v_scale = k.load(a.v_scale_ptr);
    const out_scale = k.load(a.out_scale_ptr);
    const softcap = k.load(a.softcap_ptr);
    const block_table_stride = k.load(a.block_table_stride_ptr);
    const query_stride_0 = k.load(a.query_stride_0_ptr);
    const query_stride_1 = k.load(a.query_stride_1_ptr);
    const output_stride_0 = k.load(a.output_stride_0_ptr);
    const output_stride_1 = k.load(a.output_stride_1_ptr);
    const qq_bias_stride_0 = k.load(a.qq_bias_stride_0_ptr);
    const stride_k_cache_0 = k.load(a.stride_k_cache_0_ptr);
    const stride_k_cache_1 = k.load(a.stride_k_cache_1_ptr);
    const stride_k_cache_2 = k.load(a.stride_k_cache_2_ptr);
    const stride_v_cache_0 = k.load(a.stride_v_cache_0_ptr);
    const stride_v_cache_1 = k.load(a.stride_v_cache_1_ptr);
    const stride_v_cache_2 = k.load(a.stride_v_cache_2_ptr);
    const num_seqs = k.load(a.num_seqs_ptr);

    kernelUnifiedAttention2d(
        k,
        a.output_ptr,
        a.query_ptr,
        a.key_cache_ptr,
        a.value_cache_ptr,
        a.sink_ptr,
        a.block_tables_ptr,
        a.seq_lens_ptr,
        a.alibi_slopes_ptr,
        a.qq_bias_ptr,
        scale,
        k_scale,
        v_scale,
        out_scale,
        softcap,
        block_table_stride,
        query_stride_0,
        query_stride_1,
        output_stride_0,
        output_stride_1,
        qq_bias_stride_0,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        a.query_start_len_ptr,
        num_seqs,
        config,
    );

    return k.finish(&.{}) catch |err| {
        log.err("failed to finalize kernel_unified_attention_2d_ptr TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

/// Build the TTIR for `kernel_unified_attention_3d_ptr`.
pub fn kernelUnifiedAttention3dPtr(allocator: std.mem.Allocator, config: Config3D) ![:0]const u8 {
    var spec = try Kernel.build(allocator, ctx(), "kernel_unified_attention_3d_ptr", .{
        .query_ptr = .{ .ptr = config.q_dtype },
        .key_cache_ptr = .{ .ptr = config.kv_dtype },
        .value_cache_ptr = .{ .ptr = config.kv_dtype },
        .sink_ptr = .{ .ptr = config.scale_dtype },
        .block_tables_ptr = .{ .ptr = .i32 },
        .seq_lens_ptr = .{ .ptr = .i32 },
        .alibi_slopes_ptr = .{ .ptr = config.scale_dtype },
        .qq_bias_ptr = .{ .ptr = config.scale_dtype },
        .scale_ptr = .{ .ptr = .f32 },
        .k_scale_ptr = .{ .ptr = .f32 },
        .v_scale_ptr = .{ .ptr = .f32 },
        .softcap_ptr = .{ .ptr = .f32 },
        .block_table_stride_ptr = .{ .ptr = .i64 },
        .query_stride_0_ptr = .{ .ptr = .i64 },
        .query_stride_1_ptr = .{ .ptr = .i64 },
        .qq_bias_stride_0_ptr = .{ .ptr = .i64 },
        .stride_k_cache_0_ptr = .{ .ptr = .i64 },
        .stride_k_cache_1_ptr = .{ .ptr = .i64 },
        .stride_k_cache_2_ptr = .{ .ptr = .i64 },
        .stride_v_cache_0_ptr = .{ .ptr = .i64 },
        .stride_v_cache_1_ptr = .{ .ptr = .i64 },
        .stride_v_cache_2_ptr = .{ .ptr = .i64 },
        .query_start_len_ptr = .{ .ptr = .i32 },
        .num_seqs_ptr = .{ .ptr = .i32 },
        .segm_output_ptr = .{ .ptr = .f32 },
        .segm_max_ptr = .{ .ptr = .f32 },
        .segm_expsum_ptr = .{ .ptr = .f32 },
    }, &.{});
    defer spec.deinit();
    const k = &spec.kernel;
    const a = spec.args;

    const scale = k.load(a.scale_ptr);
    const k_scale = k.load(a.k_scale_ptr);
    const v_scale = k.load(a.v_scale_ptr);
    const softcap = k.load(a.softcap_ptr);
    const block_table_stride = k.load(a.block_table_stride_ptr);
    const query_stride_0 = k.load(a.query_stride_0_ptr);
    const query_stride_1 = k.load(a.query_stride_1_ptr);
    const qq_bias_stride_0 = k.load(a.qq_bias_stride_0_ptr);
    const stride_k_cache_0 = k.load(a.stride_k_cache_0_ptr);
    const stride_k_cache_1 = k.load(a.stride_k_cache_1_ptr);
    const stride_k_cache_2 = k.load(a.stride_k_cache_2_ptr);
    const stride_v_cache_0 = k.load(a.stride_v_cache_0_ptr);
    const stride_v_cache_1 = k.load(a.stride_v_cache_1_ptr);
    const stride_v_cache_2 = k.load(a.stride_v_cache_2_ptr);
    const num_seqs = k.load(a.num_seqs_ptr);

    kernelUnifiedAttention3d(
        k,
        a.segm_output_ptr,
        a.segm_max_ptr,
        a.segm_expsum_ptr,
        a.query_ptr,
        a.key_cache_ptr,
        a.value_cache_ptr,
        a.sink_ptr,
        a.block_tables_ptr,
        a.seq_lens_ptr,
        a.alibi_slopes_ptr,
        a.qq_bias_ptr,
        scale,
        k_scale,
        v_scale,
        softcap,
        block_table_stride,
        query_stride_0,
        query_stride_1,
        qq_bias_stride_0,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        a.query_start_len_ptr,
        num_seqs,
        config,
    );

    return k.finish(&.{}) catch |err| {
        log.err("failed to finalize kernel_unified_attention_3d_ptr TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

/// Build the TTIR for `reduce_segments_ptr`.
pub fn reduceSegmentsPtr(allocator: std.mem.Allocator, config: ConfigReduce) ![:0]const u8 {
    var spec = try Kernel.build(allocator, ctx(), "reduce_segments_ptr", .{
        .segm_output_ptr = .{ .ptr = .f32 },
        .segm_max_ptr = .{ .ptr = .f32 },
        .segm_expsum_ptr = .{ .ptr = .f32 },
        .seq_lens_ptr = .{ .ptr = .i32 },
        .num_seqs_ptr = .{ .ptr = .i32 },
        .out_scale_inv_ptr = .{ .ptr = .f32 },
        .output_stride_0_ptr = .{ .ptr = .i64 },
        .output_stride_1_ptr = .{ .ptr = .i64 },
        .block_table_stride_ptr = .{ .ptr = .i64 },
        .query_start_len_ptr = .{ .ptr = .i32 },
        .output_ptr = .{ .ptr = config.o_dtype },
    }, &.{});
    defer spec.deinit();
    const k = &spec.kernel;
    const a = spec.args;

    const num_seqs = k.load(a.num_seqs_ptr);
    const out_scale_inv = k.load(a.out_scale_inv_ptr);
    const output_stride_0 = k.load(a.output_stride_0_ptr);
    const output_stride_1 = k.load(a.output_stride_1_ptr);
    const block_table_stride = k.load(a.block_table_stride_ptr);

    reduceSegments(
        k,
        a.output_ptr,
        a.segm_output_ptr,
        a.segm_max_ptr,
        a.segm_expsum_ptr,
        a.seq_lens_ptr,
        num_seqs,
        out_scale_inv,
        output_stride_0,
        output_stride_1,
        block_table_stride,
        a.query_start_len_ptr,
        config,
    );

    return k.finish(&.{}) catch |err| {
        log.err("failed to finalize reduce_segments_ptr TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}
