const std = @import("std");

const zml = @import("../../zml.zig");
const ops = zml.ops;
const tri = zml.kernel.triton;
const Builder = tri.Builder;
const DType = tri.DType;
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const Value = tri.Value;

pub const Config = struct {
    q_dtype: DType = .bf16,
    kv_dtype: DType = .bf16,
    sink_dtype: DType = .f32,
    o_dtype: DType = .bf16,
    num_query_heads: i64 = 32,
    block_size: i64 = 16,
    stride_cache_dim: i64 = 1,
    topk_count: i64 = 32,
    block_m: i64 = 16,
    rope_rank: i64 = 64,
    latent_rank: i64 = 512,
    rope_offset: i64 = 512,
    tile_size: i64 = 16,
    num_splits: i64 = 1,
    use_attn_sink: bool = false,
    all_decode: bool = false,
};

pub const Kernel2D = tri.Kernel(Config, .{
    .name = "_kernel_unified_attention_sparse_mla_2d_ptr",
    .inputs = &.{
        "query_ptr",
        "kv_cache_ptr",
        "attn_sink_ptr",
        "topk_indices_ptr",
        "scale_ptr",
        "query_stride_0_ptr",
        "query_stride_1_ptr",
        "output_stride_0_ptr",
        "output_stride_1_ptr",
        "stride_cache_0_ptr",
        "stride_cache_1_ptr",
    },
    .outputs = &.{"output"},
    .run = run2D,
});

fn run2D(b: *tri.Builder, cfg: Config) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .query_ptr = .{ .ptr = cfg.q_dtype },
        .kv_cache_ptr = .{ .ptr = cfg.kv_dtype },
        .attn_sink_ptr = .{ .ptr = cfg.sink_dtype },
        .topk_indices_ptr = .{ .ptr = .i32 },
        .scale_ptr = .{ .ptr = .f32 },
        .query_stride_0_ptr = .{ .ptr = .i64 },
        .query_stride_1_ptr = .{ .ptr = .i64 },
        .output_stride_0_ptr = .{ .ptr = .i64 },
        .output_stride_1_ptr = .{ .ptr = .i64 },
        .stride_cache_0_ptr = .{ .ptr = .i64 },
        .stride_cache_1_ptr = .{ .ptr = .i64 },
        .output_ptr = .{ .ptr = cfg.o_dtype },
    });

    const scale = b.load(a.scale_ptr);
    const query_stride_0 = b.load(a.query_stride_0_ptr);
    const query_stride_1 = b.load(a.query_stride_1_ptr);
    const output_stride_0 = b.load(a.output_stride_0_ptr);
    const output_stride_1 = b.load(a.output_stride_1_ptr);
    const stride_cache_0 = b.load(a.stride_cache_0_ptr);
    const stride_cache_1 = b.load(a.stride_cache_1_ptr);

    kernelUnifiedAttentionSparseMla(
        b,
        a.output_ptr,
        null,
        a.query_ptr,
        a.kv_cache_ptr,
        a.attn_sink_ptr,
        a.topk_indices_ptr,
        scale,
        query_stride_0,
        query_stride_1,
        output_stride_0,
        output_stride_1,
        stride_cache_0,
        stride_cache_1,
        cfg,
        false,
    );
}

pub const Kernel3D = tri.Kernel(Config, .{
    .name = "_kernel_unified_attention_sparse_mla_3d_ptr",
    .inputs = &.{
        "query_ptr",
        "kv_cache_ptr",
        "attn_sink_ptr",
        "topk_indices_ptr",
        "scale_ptr",
        "query_stride_0_ptr",
        "query_stride_1_ptr",
        "output_stride_0_ptr",
        "output_stride_1_ptr",
        "stride_cache_0_ptr",
        "stride_cache_1_ptr",
    },
    .outputs = &.{ "partial_output", "partial_lse" },
    .run = run3D,
});

fn run3D(b: *tri.Builder, cfg: Config) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .query_ptr = .{ .ptr = cfg.q_dtype },
        .kv_cache_ptr = .{ .ptr = cfg.kv_dtype },
        .attn_sink_ptr = .{ .ptr = cfg.sink_dtype },
        .topk_indices_ptr = .{ .ptr = .i32 },
        .scale_ptr = .{ .ptr = .f32 },
        .query_stride_0_ptr = .{ .ptr = .i64 },
        .query_stride_1_ptr = .{ .ptr = .i64 },
        .output_stride_0_ptr = .{ .ptr = .i64 },
        .output_stride_1_ptr = .{ .ptr = .i64 },
        .stride_cache_0_ptr = .{ .ptr = .i64 },
        .stride_cache_1_ptr = .{ .ptr = .i64 },
        .partial_output_ptr = .{ .ptr = .f32 },
        .partial_lse_ptr = .{ .ptr = .f32 },
    });

    const scale = b.load(a.scale_ptr);
    const query_stride_0 = b.load(a.query_stride_0_ptr);
    const query_stride_1 = b.load(a.query_stride_1_ptr);
    const output_stride_0 = b.load(a.output_stride_0_ptr);
    const output_stride_1 = b.load(a.output_stride_1_ptr);
    const stride_cache_0 = b.load(a.stride_cache_0_ptr);
    const stride_cache_1 = b.load(a.stride_cache_1_ptr);

    kernelUnifiedAttentionSparseMla(
        b,
        a.partial_output_ptr,
        a.partial_lse_ptr,
        a.query_ptr,
        a.kv_cache_ptr,
        a.attn_sink_ptr,
        a.topk_indices_ptr,
        scale,
        query_stride_0,
        query_stride_1,
        output_stride_0,
        output_stride_1,
        stride_cache_0,
        stride_cache_1,
        cfg,
        true,
    );
}

fn kernelUnifiedAttentionSparseMla(
    k: *Builder,
    output_ptr: Value,
    partial_lse_ptr: ?Value,
    query_ptr: Value,
    kv_cache_ptr: Value,
    attn_sink_ptr: Value,
    topk_indices_ptr: Value,
    scale: Value,
    query_stride_0: Value,
    query_stride_1: Value,
    output_stride_0: Value,
    output_stride_1: Value,
    stride_cache_0: Value,
    stride_cache_1: Value,
    config: Config,
    comptime three_d: bool,
) void {
    const BLOCK_M: i64 = config.block_m;
    const BLOCK_SIZE: i64 = config.block_size;
    const ROPE_RANK: i64 = config.rope_rank;
    const ROPE_RANK_PADDED: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(ROPE_RANK)));
    const LATENT_RANK: i64 = config.latent_rank;
    const LATENT_RANK_PADDED: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(LATENT_RANK)));
    const ROPE_OFFSET: i64 = config.rope_offset;
    const TILE_SIZE: i64 = config.tile_size;
    const NUM_QUERY_HEADS: i64 = config.num_query_heads;
    const NUM_HEAD_BLOCKS: i64 = @divTrunc(NUM_QUERY_HEADS + BLOCK_M - 1, BLOCK_M);
    const NUM_TILES: i64 = @divTrunc(config.topk_count + TILE_SIZE - 1, TILE_SIZE);
    const NUM_SPLITS: i64 = if (three_d) config.num_splits else 1;
    const TILES_PER_SPLIT: i64 = @divTrunc(NUM_TILES + NUM_SPLITS - 1, NUM_SPLITS);

    const q_block_global_idx = k.programId(.x);
    const split_idx = if (three_d) k.programId(.y) else k.liftAs(0, .i32);
    const split_tile_start = split_idx.mul(@as(i32, @intCast(NUM_TILES))).div(@as(i32, @intCast(NUM_SPLITS)));
    const split_tile_end = split_idx.add(1).mul(@as(i32, @intCast(NUM_TILES))).div(@as(i32, @intCast(NUM_SPLITS)));
    const q_ind = q_block_global_idx.div(@as(i32, @intCast(NUM_HEAD_BLOCKS)));
    const head_ind = q_block_global_idx.rem(@as(i32, @intCast(NUM_HEAD_BLOCKS)));

    const offs_h = k.arange(0, BLOCK_M, .i32).add(head_ind.mul(@as(i32, @intCast(BLOCK_M))));
    const offs_latent = k.arange(0, LATENT_RANK_PADDED, .i32);
    const latent_mask = offs_latent.lt(@as(i32, @intCast(LATENT_RANK)));
    const offs_rope_local = k.arange(0, ROPE_RANK_PADDED, .i32);
    const rope_mask = offs_rope_local.lt(@as(i32, @intCast(ROPE_RANK)));
    const offs_rope = offs_rope_local.add(@as(i32, @intCast(ROPE_OFFSET)));
    const offs_t = k.arange(0, TILE_SIZE, .i32);

    const query_offset_0 = k.splat(q_ind, &.{BLOCK_M});
    const query_offset_1 = offs_h;
    const head_mask = offs_h.lt(@as(i32, @intCast(NUM_QUERY_HEADS)));

    const qo0_2d = query_offset_0.expandDims(1).mul(query_stride_0);
    const qo1_2d = query_offset_1.expandDims(1).mul(query_stride_1);
    const q_rope_offset = qo0_2d.add(qo1_2d).add(offs_rope.expandDims(0));
    const q_rope_mask = head_mask.expandDims(1).bitAnd(rope_mask.expandDims(0));
    const Q_rope = k.loadOpts(query_ptr.addPtr(q_rope_offset), .{
        .mask = q_rope_mask,
        .other = k.zeros(&.{ BLOCK_M, ROPE_RANK_PADDED }, config.q_dtype),
        .cache_modifier = if (config.all_decode or BLOCK_M >= NUM_QUERY_HEADS) .cg else .none,
    });

    const q_latent_offset = qo0_2d.add(qo1_2d).add(offs_latent.expandDims(0));
    const q_latent_mask = head_mask.expandDims(1).bitAnd(latent_mask.expandDims(0));
    const Q_latent = k.loadOpts(query_ptr.addPtr(q_latent_offset), .{
        .mask = q_latent_mask,
        .other = k.zeros(&.{ BLOCK_M, LATENT_RANK_PADDED }, config.q_dtype),
        .cache_modifier = if (config.all_decode or BLOCK_M >= NUM_QUERY_HEADS) .cg else .none,
    });

    const m_init = k.full(&.{BLOCK_M}, -std.math.inf(f32), .f32);
    const l_init = k.full(&.{BLOCK_M}, 0.0, .f32);
    const acc_init = k.zeros(&.{ BLOCK_M, LATENT_RANK_PADDED }, .f32);

    var loop = k.openFor(0, TILES_PER_SPLIT, 1, .{ m_init, l_init, acc_init });
    {
        const t = loop.iv;
        const M = loop.carried[0];
        const L = loop.carried[1];
        const acc = loop.carried[2];

        const tile_idx = split_tile_start.add(t);
        const tile_start = tile_idx.mul(@as(i32, @intCast(TILE_SIZE)));
        const tile_offsets = tile_start.add(offs_t);
        var valid_t = tile_offsets.lt(@as(i32, @intCast(config.topk_count)))
            .bitAnd(tile_idx.lt(split_tile_end));

        const topk_row_ptr = topk_indices_ptr.addPtr(q_ind.mul(@as(i32, @intCast(config.topk_count))));
        const topk_pos = k.loadOpts(topk_row_ptr.addPtr(tile_start).addPtr(offs_t), .{
            .mask = valid_t,
            .other = k.zeros(&.{TILE_SIZE}, .i32),
        });
        valid_t = valid_t.bitAnd(topk_pos.ge(0));

        const physical_block_idx = topk_pos.div(@as(i32, @intCast(BLOCK_SIZE)));
        const slot = topk_pos.rem(@as(i32, @intCast(BLOCK_SIZE)));

        var S = k.zeros(&.{ BLOCK_M, TILE_SIZE }, .f32);

        const physical_block_idx_t = physical_block_idx.expandDims(0);
        const cache_block_offsets = physical_block_idx_t.to(.i64).mul(stride_cache_0);
        const cache_block_ptrs = kv_cache_ptr.addPtr(cache_block_offsets);
        const k_rope_dim_offsets = offs_rope.expandDims(1).mul(@as(i32, @intCast(config.stride_cache_dim)));
        const k_rope_dim_ptrs = k.broadcastTo(cache_block_ptrs, &.{ ROPE_RANK_PADDED, TILE_SIZE })
            .addPtr(k.broadcastTo(k_rope_dim_offsets, &.{ ROPE_RANK_PADDED, TILE_SIZE }));
        const slot_t = slot.expandDims(0);
        const cache_slot_offsets = slot_t.to(.i64).mul(stride_cache_1);
        const K_rope = k.loadOpts(k_rope_dim_ptrs.addPtr(k.broadcastTo(cache_slot_offsets, &.{ ROPE_RANK_PADDED, TILE_SIZE })), .{
            .mask = rope_mask.expandDims(1).bitAnd(valid_t.expandDims(0)),
            .other = k.zeros(&.{ ROPE_RANK_PADDED, TILE_SIZE }, config.kv_dtype),
            .cache_modifier = if (config.all_decode) .cg else .none,
        });
        S = S.add(scale.mul(k.dot(Q_rope, K_rope, k.zeros(&.{ BLOCK_M, TILE_SIZE }, .f32))));

        const k_latent_dim_offsets = offs_latent.expandDims(1).mul(@as(i32, @intCast(config.stride_cache_dim)));
        const k_latent_dim_ptrs = k.broadcastTo(cache_block_ptrs, &.{ LATENT_RANK_PADDED, TILE_SIZE })
            .addPtr(k.broadcastTo(k_latent_dim_offsets, &.{ LATENT_RANK_PADDED, TILE_SIZE }));
        const k_latent_mask = latent_mask.expandDims(1).bitAnd(valid_t.expandDims(0));
        const K_latent = k.loadOpts(k_latent_dim_ptrs.addPtr(k.broadcastTo(cache_slot_offsets, &.{ LATENT_RANK_PADDED, TILE_SIZE })), .{
            .mask = k_latent_mask,
            .other = k.zeros(&.{ LATENT_RANK_PADDED, TILE_SIZE }, config.kv_dtype),
            .cache_modifier = if (config.all_decode) .cg else .none,
        });

        S = S.add(scale.mul(k.dot(Q_latent, K_latent, k.zeros(&.{ BLOCK_M, TILE_SIZE }, .f32))));

        const keep_mask = head_mask.expandDims(1)
            .bitAnd(valid_t.expandDims(0));
        S = k.where(keep_mask, S, k.full(&.{ BLOCK_M, TILE_SIZE }, -std.math.inf(f32), .f32));

        var m_j = M.maximum(k.maxOpts(S, .{ .axis = 1 }));
        m_j = k.where(m_j.gt(-std.math.inf(f32)), m_j, k.full(&.{BLOCK_M}, 0.0, .f32));
        const P = k.exp(S.sub(m_j.expandDims(1)));
        const l_j = k.sumOpts(P, .{ .axis = 1 });
        const alpha = k.exp(M.sub(m_j));

        const acc_scaled = acc.mul(alpha.expandDims(1));
        const new_L = L.mul(alpha).add(l_j);

        const V_latent = k.trans(K_latent, &.{ 1, 0 });
        const new_acc = k.dot(P.to(config.kv_dtype), V_latent, acc_scaled);

        loop.yield(.{ m_j, new_L, new_acc });
    }

    const M = loop.results[0];
    var L = loop.results[1];
    var acc = loop.results[2];

    if (!three_d and config.use_attn_sink) {
        const sink_logits = k.loadOpts(attn_sink_ptr.addPtr(query_offset_1), .{
            .mask = head_mask,
            .other = k.zeros(&.{BLOCK_M}, config.sink_dtype),
        }).to(.f32);
        const sink_score = k.where(
            head_mask,
            sink_logits,
            k.full(&.{BLOCK_M}, -std.math.inf(f32), .f32),
        );
        const n_m = M.maximum(sink_score);
        const alpha = k.exp(M.sub(n_m));
        const sink_p = k.exp(sink_score.sub(n_m));

        acc = acc.mul(alpha.expandDims(1));
        L = L.mul(alpha).add(sink_p);
    }

    const has_value = L.gt(0.0);
    const safe_l = k.where(has_value, L, k.full(&.{BLOCK_M}, 1.0, .f32));
    const one_over_l = k.full(&.{ BLOCK_M, 1 }, 1.0, .f32).div(safe_l.expandDims(1));
    acc = acc.mul(k.broadcastTo(one_over_l, &.{ BLOCK_M, LATENT_RANK_PADDED }));
    acc = k.where(
        has_value.expandDims(1),
        acc,
        k.zeros(&.{ BLOCK_M, LATENT_RANK_PADDED }, .f32),
    );

    if (three_d) {
        // Compiler-managed workspace layouts are [query, head, split, value]
        // and [query, head, split], both in f32.
        const partial_out_token_stride: i64 = NUM_QUERY_HEADS * NUM_SPLITS * LATENT_RANK;
        const partial_out_head_stride: i64 = NUM_SPLITS * LATENT_RANK;
        const partial_out_offset = q_ind.to(.i64).mul(partial_out_token_stride)
            .add(offs_h.expandDims(1).to(.i64).mul(partial_out_head_stride))
            .add(split_idx.to(.i64).mul(LATENT_RANK))
            .add(offs_latent.expandDims(0).to(.i64));
        k.storeOpts(
            output_ptr.addPtr(partial_out_offset),
            acc,
            .{ .mask = head_mask.expandDims(1).bitAnd(latent_mask.expandDims(0)) },
        );

        const partial_lse_token_stride: i64 = NUM_QUERY_HEADS * NUM_SPLITS;
        const partial_lse_offset = q_ind.to(.i64).mul(partial_lse_token_stride)
            .add(offs_h.to(.i64).mul(NUM_SPLITS))
            .add(split_idx.to(.i64));
        const partial_lse = k.where(
            has_value,
            M.add(k.log(safe_l)),
            k.full(&.{BLOCK_M}, -std.math.inf(f32), .f32),
        );
        k.storeOpts(
            partial_lse_ptr.?.addPtr(partial_lse_offset),
            partial_lse,
            .{ .mask = head_mask },
        );
        return;
    }

    const output_offsets = query_offset_0.expandDims(1).mul(output_stride_0)
        .add(query_offset_1.expandDims(1).mul(output_stride_1))
        .add(offs_latent.expandDims(0));
    k.storeOpts(
        output_ptr.addPtr(output_offsets),
        acc.to(config.o_dtype),
        .{ .mask = head_mask.expandDims(1).bitAnd(latent_mask.expandDims(0)) },
    );
}

const Reduce3DPartialsCfg = struct {
    num_query_heads: i64,
    latent_rank: i64,
    num_input_splits: i64,
    num_output_splits: i64,
};

/// Reduce a large split dimension in fixed-size groups before the final
/// attention-sink reduction. Keeping the local reduction width small avoids
/// register spilling when sparse decode uses enough splits to fill a MI300X.
pub const Reduce3DPartials = tri.Kernel(Reduce3DPartialsCfg, .{
    .name = "_kernel_reduce_sparse_mla_3d_partials_ptr",
    .inputs = &.{ "input", "input_lse" },
    .outputs = &.{ "partial_output", "partial_lse" },
    .run = runReduce3DPartials,
});

fn runReduce3DPartials(b: *tri.Builder, cfg: Reduce3DPartialsCfg) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .input_ptr = .{ .ptr = .f32 },
        .input_lse_ptr = .{ .ptr = .f32 },
        .partial_output_ptr = .{ .ptr = .f32 },
        .partial_lse_ptr = .{ .ptr = .f32 },
    });

    const query_idx = b.programId(.x);
    const head_group_idx = b.programId(.y);
    const head_idx = head_group_idx.div(@as(i32, @intCast(cfg.num_output_splits)));
    const group_idx = head_group_idx.rem(@as(i32, @intCast(cfg.num_output_splits)));
    const splits_per_group = @divExact(cfg.num_input_splits, cfg.num_output_splits);
    const LATENT_RANK_PADDED: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(cfg.latent_rank)));
    const offs_s = b.arange(0, splits_per_group, .i32);
    const offs_d = b.arange(0, LATENT_RANK_PADDED, .i32);
    const latent_mask = offs_d.lt(@as(i32, @intCast(cfg.latent_rank)));
    const group_start = group_idx.to(.i64).mul(splits_per_group);

    const input_lse_token_stride = cfg.num_query_heads * cfg.num_input_splits;
    const input_lse_offset = query_idx.to(.i64).mul(input_lse_token_stride)
        .add(head_idx.to(.i64).mul(cfg.num_input_splits))
        .add(group_start)
        .add(offs_s.to(.i64));
    const input_lse = b.load(a.input_lse_ptr.addPtr(input_lse_offset));
    const overall_max = b.max(input_lse);
    const has_value = overall_max.gt(-std.math.inf(f32));
    const safe_max = b.where(has_value, overall_max, b.liftAs(0.0, .f32));
    const split_weights = b.exp(input_lse.sub(safe_max));
    const denominator = b.sumOpts(split_weights, .{ .axis = 0 });

    const input_output_token_stride = cfg.num_query_heads * cfg.num_input_splits * cfg.latent_rank;
    const input_output_head_stride = cfg.num_input_splits * cfg.latent_rank;
    const input_output_base = query_idx.to(.i64).mul(input_output_token_stride)
        .add(head_idx.to(.i64).mul(input_output_head_stride))
        .add(group_start.mul(cfg.latent_rank));
    const input_output_offset = b.broadcastTo(
        input_output_base.add(offs_s.expandDims(1).to(.i64).mul(cfg.latent_rank)),
        &.{ splits_per_group, LATENT_RANK_PADDED },
    ).add(b.broadcastTo(offs_d.expandDims(0).to(.i64), &.{ splits_per_group, LATENT_RANK_PADDED }));
    const input_output = b.loadOpts(
        a.input_ptr.addPtr(input_output_offset),
        .{
            .mask = b.broadcastTo(latent_mask.expandDims(0), &.{ splits_per_group, LATENT_RANK_PADDED }),
            .other = b.zeros(&.{ splits_per_group, LATENT_RANK_PADDED }, .f32),
        },
    );
    const numerator = b.sumOpts(input_output.mul(split_weights.expandDims(1)), .{ .axis = 0 });
    const safe_denominator = b.where(denominator.gt(0.0), denominator, b.liftAs(1.0, .f32));
    const output = b.where(
        has_value,
        numerator.div(safe_denominator),
        b.zeros(&.{LATENT_RANK_PADDED}, .f32),
    );

    const output_token_stride = cfg.num_query_heads * cfg.num_output_splits * cfg.latent_rank;
    const output_head_stride = cfg.num_output_splits * cfg.latent_rank;
    const output_offset = query_idx.to(.i64).mul(output_token_stride)
        .add(head_idx.to(.i64).mul(output_head_stride))
        .add(group_idx.to(.i64).mul(cfg.latent_rank))
        .add(offs_d.to(.i64));
    b.storeOpts(a.partial_output_ptr.addPtr(output_offset), output, .{ .mask = latent_mask });

    const output_lse_offset = query_idx.to(.i64).mul(cfg.num_query_heads * cfg.num_output_splits)
        .add(head_idx.to(.i64).mul(cfg.num_output_splits))
        .add(group_idx.to(.i64));
    const output_lse = b.where(
        has_value,
        safe_max.add(b.log(safe_denominator)),
        b.liftAs(-std.math.inf(f32), .f32),
    );
    b.store(a.partial_lse_ptr.addPtr(output_lse_offset), output_lse);
}

const Reduce3DCfg = struct {
    sink_dtype: DType = .f32,
    o_dtype: DType = .bf16,
    num_query_heads: i64 = 32,
    latent_rank: i64 = 512,
    num_splits: i64 = 1,
    use_attn_sink: bool = false,
};

pub const Reduce3D = tri.Kernel(Reduce3DCfg, .{
    .name = "_kernel_reduce_sparse_mla_3d_ptr",
    .inputs = &.{
        "partial_output_ptr",
        "partial_lse_ptr",
        "attn_sink_ptr",
        "output_stride_0_ptr",
        "output_stride_1_ptr",
    },
    .outputs = &.{"output"},
    .run = runReduce3D,
});

fn runReduce3D(b: *tri.Builder, cfg: Reduce3DCfg) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .partial_output_ptr = .{ .ptr = .f32 },
        .partial_lse_ptr = .{ .ptr = .f32 },
        .attn_sink_ptr = .{ .ptr = cfg.sink_dtype },
        .output_stride_0_ptr = .{ .ptr = .i64 },
        .output_stride_1_ptr = .{ .ptr = .i64 },
        .output_ptr = .{ .ptr = cfg.o_dtype },
    });

    const output_stride_0 = b.load(a.output_stride_0_ptr);
    const output_stride_1 = b.load(a.output_stride_1_ptr);
    const query_idx = b.programId(.x);
    const head_idx = b.programId(.y);
    const LATENT_RANK_PADDED: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(cfg.latent_rank)));
    const offs_s = b.arange(0, cfg.num_splits, .i32);
    const offs_d = b.arange(0, LATENT_RANK_PADDED, .i32);
    const latent_mask = offs_d.lt(@as(i32, @intCast(cfg.latent_rank)));

    const partial_lse_token_stride: i64 = cfg.num_query_heads * cfg.num_splits;
    const partial_lse_offset = query_idx.to(.i64).mul(partial_lse_token_stride)
        .add(head_idx.to(.i64).mul(cfg.num_splits))
        .add(offs_s.to(.i64));
    const partial_lse = b.load(a.partial_lse_ptr.addPtr(partial_lse_offset));

    const sink_score = if (cfg.use_attn_sink)
        b.load(a.attn_sink_ptr.addPtr(head_idx)).to(.f32)
    else
        b.liftAs(-std.math.inf(f32), .f32);
    const overall_max = b.max(partial_lse).maximum(sink_score);
    const has_value = overall_max.gt(-std.math.inf(f32));
    const safe_max = b.where(has_value, overall_max, b.liftAs(0.0, .f32));
    const split_weights = b.exp(partial_lse.sub(safe_max));
    const sink_weight = if (cfg.use_attn_sink)
        b.exp(sink_score.sub(safe_max))
    else
        b.liftAs(0.0, .f32);
    const denominator = b.sumOpts(split_weights, .{ .axis = 0 }).add(sink_weight);

    const partial_out_token_stride: i64 = cfg.num_query_heads * cfg.num_splits * cfg.latent_rank;
    const partial_out_head_stride: i64 = cfg.num_splits * cfg.latent_rank;
    const partial_out_base = query_idx.to(.i64).mul(partial_out_token_stride)
        .add(head_idx.to(.i64).mul(partial_out_head_stride));
    const partial_out_offset = b.broadcastTo(
        partial_out_base.add(offs_s.expandDims(1).to(.i64).mul(cfg.latent_rank)),
        &.{ cfg.num_splits, LATENT_RANK_PADDED },
    ).add(b.broadcastTo(offs_d.expandDims(0).to(.i64), &.{ cfg.num_splits, LATENT_RANK_PADDED }));
    const partial_output = b.loadOpts(
        a.partial_output_ptr.addPtr(partial_out_offset),
        .{
            .mask = b.broadcastTo(latent_mask.expandDims(0), &.{ cfg.num_splits, LATENT_RANK_PADDED }),
            .other = b.zeros(&.{ cfg.num_splits, LATENT_RANK_PADDED }, .f32),
        },
    );
    const weighted_output = partial_output.mul(split_weights.expandDims(1));
    const numerator = b.sumOpts(weighted_output, .{ .axis = 0 });
    const safe_denominator = b.where(
        denominator.gt(0.0),
        denominator,
        b.liftAs(1.0, .f32),
    );
    const output = b.where(
        has_value,
        numerator.div(safe_denominator),
        b.zeros(&.{LATENT_RANK_PADDED}, .f32),
    );

    const output_offset = query_idx.to(.i64).mul(output_stride_0)
        .add(head_idx.to(.i64).mul(output_stride_1))
        .add(offs_d.to(.i64));
    b.storeOpts(
        a.output_ptr.addPtr(output_offset),
        output.to(cfg.o_dtype),
        .{ .mask = latent_mask },
    );
}
