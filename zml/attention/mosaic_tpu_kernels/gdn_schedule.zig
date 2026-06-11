//! Schedule_table builder for P5 (`recurrent_scan`). Port of tpu-inference's
//! `compute_schedule_v2.py`. Builds the `[safe_max_blocks, 11 + 3*alignment]`
//! i32 table the P5 kernel reads per grid step. Columns documented in
//! `compute_schedule_v2.py:220-256`. Pure `zml.Tensor` API — no Mosaic builder
//! dependency; lives in the high-level attention kernels alongside
//! `ragged_attention.zig`.

const std = @import("std");
const zml = @import("../../zml.zig");
const Tensor = zml.Tensor;
const Shape = zml.Shape;

/// Output of the schedule builder.
pub const ScheduleResult = struct {
    /// `[safe_max_blocks, 11 + 3*alignment]` i32 schedule table.
    table: Tensor,
    /// Scalar i32: `max(total_prefill_blocks, num_decode_batches)` —
    /// the grid bound the P5 `pl.when(num_steps > 0)` checks against.
    total_blocks: Tensor,
};

fn const_i32(v: anytype) Tensor {
    return Tensor.scalar(@as(i32, @intCast(v)), .i32);
}

/// Build the schedule_table. See module docstring.
///
/// Static (comptime) args:
///   `safe_max_blocks`, `num_seqs`, `chunk_size`, `BT`, `alignment`.
/// Runtime args:
///   `query_start_loc` `[num_seqs+1]` i32
///   `decode_tokens` scalar i32
///   `num_valid_seqs` scalar i32
pub fn buildScheduleTable(
    query_start_loc_in: Tensor,
    decode_tokens: Tensor,
    num_valid_seqs: Tensor,
    safe_max_blocks: i64,
    num_seqs: i64,
    chunk_size: i64,
    BT: i64,
    comptime alignment: i64,
) ScheduleResult {
    // Normalize input tags so the rest of the function can use `.r` uniformly.
    // Callers commonly pass `query_start_loc` with tag `.seq` (llmd's
    // `withTags(.{.seq})` upstream). `num_valid_seqs` may arrive as a `(1,)`
    // tensor with tag `.b` from llmd; reshape to a scalar for gather indexing.
    const query_start_loc = query_start_loc_in.rename(.{ .seq = .r });
    const nvs_scalar = num_valid_seqs.reshape(Shape.init(.{}, .i32));
    const r_shape = Shape.init(.{ .r = num_seqs }, .i32);
    const b_shape = Shape.init(.{ .b = safe_max_blocks }, .i32);
    const ba_shape = Shape.init(.{ .b = safe_max_blocks, .a = alignment }, .i32);

    // ── num_decode_batches = ceil(decode_tokens / BT)
    const num_decode_batches = decode_tokens
        .add(const_i32(BT - 1).broad(decode_tokens.shape()))
        .div(const_i32(BT).broad(decode_tokens.shape()));

    // ── per-seq scalars over r ──
    const r_idx = Tensor.iota(r_shape, .r);
    const is_last_seq = r_idx.cmp(.EQ, const_i32(num_seqs - 1).broad(r_shape));

    const seq_start = query_start_loc.slice1d(.r, .{ .end = num_seqs });
    const seq_end = query_start_loc.slice1d(.r, .{ .start = 1 });

    // prev_seq_end[r] = r > 0 ? seq_end[r-1] : 0 — use slice + pad-low(1).
    // For num_seqs == 1 the slice would be `[0..0]` (empty), which trips the
    // slice1d assert; the answer is just `[0]` regardless.
    const prev_seq_end = if (num_seqs <= 1)
        Tensor.zeroes(r_shape)
    else
        seq_end.slice1d(.r, .{ .end = num_seqs - 1 }).pad(@as(i32, 0), .{ .r = Tensor.Pad{ .low = 1 } });

    // effective_start = (prev_seq_end % alignment != 0) ?
    //                   (prev_seq_end / alignment) * alignment + alignment : prev_seq_end
    const align_t = const_i32(alignment);
    const prev_mod = modI32(prev_seq_end, alignment);
    const prev_round_up = prev_seq_end.div(align_t.broad(prev_seq_end.shape()))
        .mul(align_t.broad(prev_seq_end.shape()))
        .add(align_t.broad(prev_seq_end.shape()));
    const prev_mod_nz = prev_mod.cmp(.NE, const_i32(0).broad(prev_mod.shape()));
    const effective_start = prev_mod_nz.select(prev_round_up, prev_seq_end);

    // is_decode_boundary = prev_seq_end == decode_tokens  (broadcast scalar)
    const is_decode_boundary = prev_seq_end.cmp(.EQ, decode_tokens.broad(prev_seq_end.shape()));

    // is_swallowed = (effective_start >= seq_end) & !is_decode_boundary
    const eff_ge_end = effective_start.cmp(.GE, seq_end);
    const is_swallowed = boolAnd(eff_ge_end, boolNot(is_decode_boundary));

    // next_aligned_start = (seq_end / alignment) * alignment
    const next_aligned_start = seq_end.div(align_t.broad(seq_end.shape()))
        .mul(align_t.broad(seq_end.shape()));

    // needs_transition = (seq_end % align != 0) & !is_last_seq & !is_swallowed
    const seq_end_mod_nz = modI32(seq_end, alignment).cmp(.NE, const_i32(0).broad(seq_end.shape()));
    const needs_transition = boolAnd(seq_end_mod_nz, boolAnd(boolNot(is_last_seq), boolNot(is_swallowed)));

    // needs_start_transition = (prev_seq_end % align != 0) & !is_swallowed & is_decode_boundary
    const needs_start_transition = boolAnd(prev_mod_nz, boolAnd(boolNot(is_swallowed), is_decode_boundary));

    // effective_end = max(effective_start, where(needs_transition, next_aligned_start, seq_end))
    const eff_end_pre = needs_transition.select(next_aligned_start, seq_end);
    const effective_end = eff_end_pre.maximum(effective_start);

    // num_regular_blocks = ceil((effective_end - effective_start) / chunk_size)
    const span = effective_end.sub(effective_start);
    const num_regular_blocks = span.add(const_i32(chunk_size - 1).broad(span.shape()))
        .div(const_i32(chunk_size).broad(span.shape()));

    // total_blocks_per_seq
    const tot_with_trans = num_regular_blocks
        .add(needs_transition.convert(.i32))
        .add(needs_start_transition.convert(.i32));
    const tot_after_swallow = is_swallowed.select(const_i32(0).broad(tot_with_trans.shape()), tot_with_trans);
    const is_pure_decode = seq_end.cmp(.LE, decode_tokens.broad(seq_end.shape()));
    const total_blocks_per_seq = is_pure_decode.select(const_i32(0).broad(tot_after_swallow.shape()), tot_after_swallow);

    // base_idx = cumsum(total_blocks_per_seq) - total_blocks_per_seq
    const cum = total_blocks_per_seq.cumulativeSum(.r);
    const base_idx = cum.sub(total_blocks_per_seq);
    const total_prefill_blocks = total_blocks_per_seq.sum(.r).squeeze(.r);

    // ── per-block ──
    const b_idx = Tensor.iota(b_shape, .b);
    const prefill_valid_mask = b_idx.cmp(.LT, total_prefill_blocks.broad(b_shape));

    // r_for_block: sum(b_idx[:, None] >= base_idx[None, :]) - 1, clamped.
    const b_idx_2d_shape = Shape.init(.{ .b = safe_max_blocks, .r = num_seqs }, .i32);
    const b_idx_2d = b_idx.insertAxes(1, .{.r}).broad(b_idx_2d_shape);
    const base_idx_2d = base_idx.insertAxes(0, .{.b}).broad(b_idx_2d_shape);
    const r_for_block_pre = b_idx_2d.cmp(.GE, base_idx_2d).convert(.i32).sum(.r).squeeze(.r)
        .sub(const_i32(1).broad(b_shape));
    const r_for_block = r_for_block_pre
        .maximum(const_i32(0).broad(b_shape))
        .minimum(const_i32(num_seqs - 1).broad(b_shape));

    // local_b = b_idx - base_idx[r_for_block]
    const base_at_r = base_idx.gather(.{ .r = r_for_block }, .{});
    const local_b = b_idx.sub(base_at_r);

    // is_start_trans
    const needs_start_at_r = needs_start_transition.convert(.i32).gather(.{ .r = r_for_block }, .{});
    const needs_start_at_r_bool = needs_start_at_r.cmp(.NE, const_i32(0).broad(b_shape));
    const local_b_zero = local_b.cmp(.EQ, const_i32(0).broad(b_shape));
    const is_start_trans = boolAnd(needs_start_at_r_bool, local_b_zero);

    // adj_local_b = where(needs_start_at_r != 0, local_b - 1, local_b)
    const adj_local_b = needs_start_at_r_bool.select(
        local_b.sub(const_i32(1).broad(b_shape)),
        local_b,
    );

    // is_end_trans
    const needs_trans_at_r = needs_transition.convert(.i32).gather(.{ .r = r_for_block }, .{});
    const needs_trans_at_r_bool = needs_trans_at_r.cmp(.NE, const_i32(0).broad(b_shape));
    const nrb_at_r = num_regular_blocks.gather(.{ .r = r_for_block }, .{});
    const is_end_trans = boolAnd(needs_trans_at_r_bool, adj_local_b.cmp(.EQ, nrb_at_r));

    // offsets / counts
    const eff_start_at_r = effective_start.gather(.{ .r = r_for_block }, .{});
    const eff_end_at_r = effective_end.gather(.{ .r = r_for_block }, .{});
    const reg_offset = eff_start_at_r.add(adj_local_b.mul(const_i32(chunk_size).broad(b_shape)));
    const reg_count = const_i32(chunk_size).broad(b_shape)
        .minimum(eff_end_at_r.sub(reg_offset));
    const trans_offset = next_aligned_start.gather(.{ .r = r_for_block }, .{});

    const block_offset_inner = is_end_trans.select(trans_offset, reg_offset);
    const start_trans_offset = seq_start.gather(.{ .r = r_for_block }, .{})
        .div(align_t.broad(b_shape))
        .mul(align_t.broad(b_shape));
    const block_offset_pre = is_start_trans.select(start_trans_offset, block_offset_inner);

    const seq_start_at_r = seq_start.gather(.{ .r = r_for_block }, .{});
    const start_trans_count = eff_start_at_r.sub(seq_start_at_r);
    const block_count_inner = is_end_trans.select(align_t.broad(b_shape), reg_count);
    const block_count_pre = is_start_trans.select(start_trans_count, block_count_inner);

    const is_trans_block_pre = boolOr(is_start_trans, is_end_trans);

    // Mask invalid rows.
    const zero_b = const_i32(0).broad(b_shape);
    const block_offset = prefill_valid_mask.select(block_offset_pre, zero_b);
    const r_for_block_masked = prefill_valid_mask.select(r_for_block, zero_b);
    const block_count = prefill_valid_mask.select(block_count_pre, zero_b);
    const is_trans_block = boolAnd(is_trans_block_pre, prefill_valid_mask);

    const seq_start_at_r2 = seq_start.gather(.{ .r = r_for_block_masked }, .{});
    const seq_end_at_r2 = seq_end.gather(.{ .r = r_for_block_masked }, .{});
    const block_is_first = boolAnd(block_offset.cmp(.LE, seq_start_at_r2), prefill_valid_mask);
    const block_is_last = boolAnd(block_offset.add(block_count).cmp(.GE, seq_end_at_r2), prefill_valid_mask);

    // ── sublane fields (Pallas's `compute_schedule_v2.py:167-189`) ──
    // For each (block, sublane) pair, identify which request its token belongs
    // to (`t_reqs`), and whether that token is the first / last of its request
    // (`is_first_tok` / `is_last_tok`). The kernel reads these in transition
    // blocks (`gdn_prefill.zig:1019-1082`) to drive per-token state load/save
    // logic — without correct values, all sublane tokens are attributed to
    // request 0, corrupting state across requests in batched inference.
    const sub_iota = Tensor.iota(Shape.init(.{ .a = alignment }, .i32), .a);
    const block_offset_2d = block_offset.insertAxes(1, .{.a}).broad(ba_shape);
    const sub_iota_2d = sub_iota.insertAxes(0, .{.b}).broad(ba_shape);
    const glob_idxs = block_offset_2d.add(sub_iota_2d);

    // `num_tokens = query_start_loc[num_valid_seqs]` — actual active token
    // count (vs the bucket-size last entry). Used both for the valid-mask and
    // as the clamp value below.
    const num_tokens_tensor = query_start_loc.gather(.{ .r = nvs_scalar }, .{});

    // `valid_loc_mask = arange(num_seqs+1) <= num_valid_seqs` — and clamp
    // padding positions to `last_valid_loc` so they don't fabricate request
    // boundaries.
    const qsl_shape = Shape.init(.{ .r = num_seqs + 1 }, .i32);
    const qsl_iota = Tensor.iota(qsl_shape, .r);
    const nvs_bc_qsl = nvs_scalar.broad(qsl_shape);
    const valid_loc_mask = qsl_iota.cmp(.LE, nvs_bc_qsl);
    const last_valid_loc_bc = num_tokens_tensor.broad(qsl_shape);
    const fixed_qsl = valid_loc_mask.select(query_start_loc, last_valid_loc_bc);

    // `valid_mask = glob_idxs < num_tokens` — sublane positions past the last
    // active token are invalid (they'll get clamped to `last_valid_seq` below).
    const num_tokens_bc_ba = num_tokens_tensor.broad(ba_shape);
    const valid_mask = glob_idxs.cmp(.LT, num_tokens_bc_ba);

    // `t_reqs = sum(glob_idxs[:, :, None] >= fixed_qsl[None, None, :], -1) - 1`
    // For each (b, a), counts how many fixed_qsl[r] are <= glob_idxs[b, a]
    // (monotonic so this is the index of the segment containing the token).
    const bar_shape = Shape.init(.{ .b = safe_max_blocks, .a = alignment, .r = num_seqs + 1 }, .i32);
    const glob_3d = glob_idxs.insertAxes(2, .{.r}).broad(bar_shape);
    const fixed_qsl_3d = fixed_qsl.insertAxes(0, .{ .b, .a }).broad(bar_shape);
    const t_reqs_raw = glob_3d.cmp(.GE, fixed_qsl_3d).convert(.i32).sum(.r).squeeze(.r)
        .sub(const_i32(1).broad(ba_shape));

    // `last_valid_seq = max(where(total_blocks_per_seq > 0, arange(num_seqs), -1))`
    // — used as fallback for sublanes past `num_tokens` (so they map to a real
    // request slot, not -1 or 0).
    const tbps_pos = total_blocks_per_seq.cmp(.GT, const_i32(0).broad(r_shape));
    const r_iota_for_last = Tensor.iota(r_shape, .r);
    const neg_one_r = const_i32(-1).broad(r_shape);
    const last_valid_seq = tbps_pos.select(r_iota_for_last, neg_one_r).max(.r).squeeze(.r);

    // Apply mask + clamp to `[0, num_seqs-1]`.
    const last_valid_seq_bc = last_valid_seq.broad(ba_shape);
    const t_reqs_masked = valid_mask.select(t_reqs_raw, last_valid_seq_bc);
    const t_reqs = t_reqs_masked
        .maximum(const_i32(0).broad(ba_shape))
        .minimum(const_i32(num_seqs - 1).broad(ba_shape));

    // `is_first_tok = (glob_idxs == query_start_loc[t_reqs])`
    // `is_last_tok  = (glob_idxs == query_start_loc[t_reqs + 1] - 1)`
    // Both gathers run over `query_start_loc` (NOT `fixed_qsl`) — Pallas uses
    // the raw value here; the clamp via `t_reqs ∈ [0, num_seqs-1]` already
    // restricts the gather to valid positions.
    const qsl_at_treq = query_start_loc.gather(.{ .r = t_reqs }, .{});
    const t_reqs_plus1 = t_reqs.add(const_i32(1).broad(ba_shape));
    const qsl_at_treq_plus1 = query_start_loc.gather(.{ .r = t_reqs_plus1 }, .{});
    const is_first_tok_bool = glob_idxs.cmp(.EQ, qsl_at_treq);
    const last_pos = qsl_at_treq_plus1.sub(const_i32(1).broad(ba_shape));
    const is_last_tok_bool = glob_idxs.cmp(.EQ, last_pos);
    const is_first_tok_raw = is_first_tok_bool.convert(.i32);
    const is_last_tok_raw = is_last_tok_bool.convert(.i32);

    // Zero the sublane fields on invalid rows (Pallas `compute_schedule_v2.py:213-215`
    // — `jnp.where(prefill_valid_mask[:, None], x, 0)`). Without this, padding
    // rows in `t_reqs` still hold the speculative request IDs computed from the
    // zeroed `block_offset`, which the kernel would read as "every padding-row
    // sublane belongs to request 0..k" and emit ghost state writes.
    const prefill_valid_mask_2d = prefill_valid_mask.insertAxes(1, .{.a}).broad(ba_shape);
    const t_reqs_final = prefill_valid_mask_2d.select(t_reqs, const_i32(0).broad(ba_shape));
    const is_first_tok = prefill_valid_mask_2d.select(is_first_tok_raw, const_i32(0).broad(ba_shape));
    const is_last_tok = prefill_valid_mask_2d.select(is_last_tok_raw, const_i32(0).broad(ba_shape));

    // ── decode metadata ──
    const num_dbatch_b = num_decode_batches.broad(b_shape);
    const decode_valid_mask = b_idx.cmp(.LT, num_dbatch_b);
    const ndb_minus_1 = num_decode_batches.sub(const_i32(1).broad(num_decode_batches.shape()));
    const dbi_pre = ndb_minus_1.broad(b_shape).sub(b_idx);
    const decode_batch_idx = decode_valid_mask.select(dbi_pre, zero_b);
    const decode_offsets = decode_batch_idx.mul(const_i32(BT).broad(b_shape));
    const decode_req_ids = decode_offsets;
    const decode_counts_pre = const_i32(BT).broad(b_shape)
        .minimum(decode_tokens.broad(b_shape).sub(decode_offsets));
    const decode_counts = decode_valid_mask.select(decode_counts_pre, zero_b);

    // ── assemble [safe_max_blocks, 11 + 3*alignment] ──
    // Stack has a 32-tensor cap; we use insertAxes(.col) per column then
    // concatenate along .col (which supports arbitrary arity).
    var cols_buf: [11 + 3 * 16]Tensor = undefined;
    cols_buf[0] = prefill_valid_mask.convert(.i32);
    cols_buf[1] = block_offset;
    cols_buf[2] = r_for_block_masked;
    cols_buf[3] = block_count;
    cols_buf[4] = decode_valid_mask.convert(.i32);
    cols_buf[5] = decode_offsets;
    cols_buf[6] = decode_req_ids;
    cols_buf[7] = decode_counts;
    cols_buf[8] = block_is_last.convert(.i32);
    cols_buf[9] = block_is_first.convert(.i32);
    cols_buf[10] = is_trans_block.convert(.i32);
    inline for (0..alignment) |i| {
        cols_buf[11 + i] = t_reqs_final.slice1d(.a, .{ .start = i, .end = i + 1 }).squeeze(.a);
        cols_buf[11 + alignment + i] = is_first_tok.slice1d(.a, .{ .start = i, .end = i + 1 }).squeeze(.a);
        cols_buf[11 + 2 * alignment + i] = is_last_tok.slice1d(.a, .{ .start = i, .end = i + 1 }).squeeze(.a);
    }
    // Add a singleton `.col` axis to each (Pallas emits `stablehlo.broadcast_in_dim
    // dims=[0]` for `x[:, None]`; mirror with `broad` so the IR diff against
    // tpu-inference's `compute_schedule_table_v2` doesn't drown in `reshape`
    // ↔ `broadcast_in_dim` noise on every column).
    var cols2d_buf: [11 + 3 * 16]Tensor = undefined;
    const ncols = 11 + 3 * alignment;
    const col_shape = Shape.init(.{ .b = safe_max_blocks, .col = 1 }, .i32);
    inline for (0..(11 + 3 * 16)) |i| {
        if (i < ncols) cols2d_buf[i] = cols_buf[i].broad(col_shape);
    }
    // Pallas's concat grouping (compute_schedule_v2 lowers to 16+16+3=35 along
    // dim=1). Mirror to keep the trailing `concatenate` chain byte-identical.
    const group_a_end: usize = 16;
    const group_b_end: usize = 32;
    const grp_a = Tensor.concatenate(cols2d_buf[0..group_a_end], .col);
    const grp_b = Tensor.concatenate(cols2d_buf[group_a_end..group_b_end], .col);
    const grp_c = Tensor.concatenate(cols2d_buf[group_b_end..ncols], .col);
    const final_table = Tensor.concatenate(&.{ grp_a, grp_b, grp_c }, .col);
    const total_blocks = total_prefill_blocks.maximum(num_decode_batches.reshape(total_prefill_blocks.shape()));

    return .{ .table = final_table, .total_blocks = total_blocks };
}

// ── small helpers ────────────────────────────────────────────────────────

fn boolAnd(a: Tensor, b: Tensor) Tensor {
    return a.logical(.AND, b);
}
fn boolOr(a: Tensor, b: Tensor) Tensor {
    return a.logical(.OR, b);
}
fn boolNot(a: Tensor) Tensor {
    return a.not();
}
fn modI32(a: Tensor, comptime m: i64) Tensor {
    return a.remainder(const_i32(m).broad(a.shape()));
}
