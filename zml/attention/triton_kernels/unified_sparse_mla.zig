const std = @import("std");

const zml = @import("../../zml.zig");
const ops = zml.ops;
const tri = zml.kernel.triton;
const Builder = tri.Builder;
const DType = tri.DType;
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const Value = tri.Value;

const Cfg = struct {
    q_dtype: DType = .bf16,
    kv_dtype: DType = .bf16,
    sink_dtype: DType = .f32,
    o_dtype: DType = .bf16,
    num_query_heads: i64 = 32,
    num_queries_per_kv: i64 = 32,
    block_size: i64 = 16,
    stride_k_cache_3: i64 = 1,
    stride_v_cache_3: i64 = 1,
    topk_count: i64 = 32,
    block_m: i64 = 16,
    rope_rank: i64 = 64,
    qk_lora_rank: i64 = 512,
    kv_lora_rank: i64 = 512,
    rope_offset: i64 = 512,
    value_rank: i64 = 512,
    tile_size: i64 = 16,
    use_attn_sink: bool = false,
    all_decode: bool = false,
};

pub const Kernel = tri.Kernel(Cfg, .{
    .name = "_kernel_unified_attention_sparse_mla_2d_ptr",
    .inputs = &.{
        "query_ptr",
        "key_cache_ptr",
        "value_cache_ptr",
        "attn_sink_ptr",
        "block_tables_ptr",
        "topk_indices_ptr",
        "seq_lens_ptr",
        "scale_ptr",
        "block_table_stride_ptr",
        "query_stride_0_ptr",
        "query_stride_1_ptr",
        "output_stride_0_ptr",
        "output_stride_1_ptr",
        "stride_k_cache_0_ptr",
        "stride_k_cache_1_ptr",
        "stride_k_cache_2_ptr",
        "stride_v_cache_0_ptr",
        "stride_v_cache_1_ptr",
        "stride_v_cache_2_ptr",
        "query_start_len_ptr",
        "num_seqs_ptr",
    },
    .outputs = &.{"output"},
    .run = run,
});

fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .query_ptr = .{ .ptr = cfg.q_dtype },
        .key_cache_ptr = .{ .ptr = cfg.kv_dtype },
        .value_cache_ptr = .{ .ptr = cfg.kv_dtype },
        .attn_sink_ptr = .{ .ptr = cfg.sink_dtype },
        .block_tables_ptr = .{ .ptr = .i32 },
        .topk_indices_ptr = .{ .ptr = .i32 },
        .seq_lens_ptr = .{ .ptr = .i32 },
        .scale_ptr = .{ .ptr = .f32 },
        .block_table_stride_ptr = .{ .ptr = .i64 },
        .query_stride_0_ptr = .{ .ptr = .i64 },
        .query_stride_1_ptr = .{ .ptr = .i64 },
        .output_stride_0_ptr = .{ .ptr = .i64 },
        .output_stride_1_ptr = .{ .ptr = .i64 },
        .stride_k_cache_0_ptr = .{ .ptr = .i64 },
        .stride_k_cache_1_ptr = .{ .ptr = .i64 },
        .stride_k_cache_2_ptr = .{ .ptr = .i64 },
        .stride_v_cache_0_ptr = .{ .ptr = .i64 },
        .stride_v_cache_1_ptr = .{ .ptr = .i64 },
        .stride_v_cache_2_ptr = .{ .ptr = .i64 },
        .query_start_len_ptr = .{ .ptr = .i32 },
        .num_seqs_ptr = .{ .ptr = .i32 },
        .output_ptr = .{ .ptr = cfg.o_dtype },
    });

    const scale = b.load(a.scale_ptr);
    const query_stride_0 = b.load(a.query_stride_0_ptr);
    const query_stride_1 = b.load(a.query_stride_1_ptr);
    const output_stride_0 = b.load(a.output_stride_0_ptr);
    const output_stride_1 = b.load(a.output_stride_1_ptr);
    const stride_k_cache_0 = b.load(a.stride_k_cache_0_ptr);
    const stride_k_cache_1 = b.load(a.stride_k_cache_1_ptr);
    const stride_v_cache_0 = b.load(a.stride_v_cache_0_ptr);
    const stride_v_cache_1 = b.load(a.stride_v_cache_1_ptr);
    const num_seqs = b.load(a.num_seqs_ptr);

    kernelUnifiedAttentionSparseMla2d(
        b,
        a.output_ptr,
        a.query_ptr,
        a.key_cache_ptr,
        a.value_cache_ptr,
        a.attn_sink_ptr,
        a.topk_indices_ptr,
        scale,
        query_stride_0,
        query_stride_1,
        output_stride_0,
        output_stride_1,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_v_cache_0,
        stride_v_cache_1,
        a.query_start_len_ptr,
        num_seqs,
        cfg,
    );
}

fn findSeqIdx(
    k: *Builder,
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

fn kernelUnifiedAttentionSparseMla2d(
    k: *Builder,
    output_ptr: Value,
    query_ptr: Value,
    key_cache_ptr: Value,
    value_cache_ptr: Value,
    attn_sink_ptr: Value,
    topk_indices_ptr: Value,
    scale: Value,
    query_stride_0: Value,
    query_stride_1: Value,
    output_stride_0: Value,
    output_stride_1: Value,
    stride_k_cache_0: Value,
    stride_k_cache_1: Value,
    stride_v_cache_0: Value,
    stride_v_cache_1: Value,
    query_start_len_ptr: Value,
    num_seqs: Value,
    config: Cfg,
) void {
    const BLOCK_Q: i64 = 1;
    const BLOCK_M: i64 = config.block_m;
    const BLOCK_SIZE: i64 = config.block_size;
    const ROPE_RANK: i64 = config.rope_rank;
    const QK_LORA_RANK: i64 = config.qk_lora_rank;
    const KV_LORA_RANK: i64 = config.kv_lora_rank;
    const ROPE_OFFSET: i64 = config.rope_offset;
    const VALUE_RANK: i64 = config.value_rank;
    const TILE_SIZE: i64 = config.tile_size;
    const NUM_QUERIES_PER_KV: i64 = config.num_queries_per_kv;
    const NUM_QUERY_HEADS: i64 = config.num_query_heads;
    const NUM_HEAD_BLOCKS: i64 = @divTrunc(NUM_QUERY_HEADS, BLOCK_M);
    const NUM_TILES: i64 = @divTrunc(config.topk_count + TILE_SIZE - 1, TILE_SIZE);

    const q_block_global_idx = k.programId(.x);
    const q_ind = q_block_global_idx.div(@as(i32, @intCast(NUM_HEAD_BLOCKS)));
    const head_ind = q_block_global_idx.rem(@as(i32, @intCast(NUM_HEAD_BLOCKS)));
    const seq_idx = findSeqIdx(k, query_start_len_ptr, q_ind, num_seqs, BLOCK_Q, false);
    const q_block_start_idx = k.load(query_start_len_ptr.addPtr(seq_idx));

    const q_block_local_idx = q_ind.sub(q_block_start_idx);
    const cur_batch_in_all_start_index = k.load(query_start_len_ptr.addPtr(seq_idx));
    const cur_batch_in_all_stop_index = k.load(query_start_len_ptr.addPtr(seq_idx).addPtr(1));
    const cur_batch_query_len = cur_batch_in_all_stop_index.sub(cur_batch_in_all_start_index);

    k.returnIf(q_block_local_idx.ge(cur_batch_query_len), .{});

    const offs_m = k.arange(0, BLOCK_M, .i32).add(head_ind.mul(@as(i32, @intCast(BLOCK_M))));
    const offs_lora = k.arange(0, KV_LORA_RANK, .i32);
    const offs_rope = k.arange(ROPE_OFFSET, ROPE_OFFSET + ROPE_RANK, .i32);
    const offs_value = k.arange(0, VALUE_RANK, .i32);
    const offs_t = k.arange(0, TILE_SIZE, .i32);

    const query_pos = q_block_local_idx.add(offs_m.div(@as(i32, @intCast(NUM_QUERIES_PER_KV))));

    const query_offset_0 = cur_batch_in_all_start_index.add(query_pos);
    const query_offset_1 = offs_m.rem(@as(i32, @intCast(NUM_QUERIES_PER_KV)));

    const query_mask_0 = query_pos.lt(cur_batch_query_len);
    const query_mask_1 = query_offset_1.lt(@as(i32, @intCast(NUM_QUERY_HEADS)));

    const qo0_2d = query_offset_0.expandDims(1).mul(query_stride_0);
    const qo1_2d = query_offset_1.expandDims(1).mul(query_stride_1);
    const q_rope_offset = qo0_2d.add(qo1_2d).add(offs_rope.expandDims(0));
    const q_mask = query_mask_0.expandDims(1).bitAnd(query_mask_1.expandDims(1));
    const Q_rope = k.loadOpts(query_ptr.addPtr(q_rope_offset), .{
        .mask = q_mask,
        .other = k.zeros(&.{ BLOCK_M, ROPE_RANK }, config.q_dtype),
        .cache_modifier = if (config.all_decode or BLOCK_M >= NUM_QUERY_HEADS) .cg else .none,
    });

    const q_lora_offset = qo0_2d.add(qo1_2d).add(offs_lora.expandDims(0));
    const q_lora_mask = q_mask.bitAnd(offs_lora.lt(@as(i32, @intCast(QK_LORA_RANK))).expandDims(0));
    const Q_lora = k.loadOpts(query_ptr.addPtr(q_lora_offset), .{
        .mask = q_lora_mask,
        .other = k.zeros(&.{ BLOCK_M, KV_LORA_RANK }, config.q_dtype),
        .cache_modifier = if (config.all_decode or BLOCK_M >= NUM_QUERY_HEADS) .cg else .none,
    });

    const m_init = k.full(&.{BLOCK_M}, -std.math.inf(f32), .f32);
    const l_init = k.full(&.{BLOCK_M}, 1.0, .f32);
    const acc_init = k.zeros(&.{ BLOCK_M, VALUE_RANK }, .f32);

    var loop = k.openFor(0, NUM_TILES, 1, .{ m_init, l_init, acc_init });
    {
        const t = loop.iv;
        const M = loop.carried[0];
        const L = loop.carried[1];
        const acc = loop.carried[2];

        const tile_start = t.mul(@as(i32, @intCast(TILE_SIZE)));
        const tile_offsets = tile_start.add(offs_t);
        var valid_t = tile_offsets.lt(@as(i32, @intCast(config.topk_count)));

        const topk_row_ptr = topk_indices_ptr.addPtr(q_ind.mul(@as(i32, @intCast(config.topk_count))));
        const topk_pos = k.loadOpts(topk_row_ptr.addPtr(tile_start).addPtr(offs_t), .{
            .mask = valid_t,
            .other = k.zeros(&.{TILE_SIZE}, .i32),
        });
        valid_t = valid_t.bitAnd(topk_pos.ne(-1));

        const physical_block_idx = topk_pos.div(@as(i32, @intCast(BLOCK_SIZE)));
        const slot = topk_pos.rem(@as(i32, @intCast(BLOCK_SIZE)));

        var S = k.zeros(&.{ BLOCK_M, TILE_SIZE }, .f32);

        const physical_block_idx_t = physical_block_idx.expandDims(0);
        const key_block_offsets = physical_block_idx_t.to(.i64).mul(stride_k_cache_0);
        const key_block_ptrs = key_cache_ptr.addPtr(key_block_offsets);
        const k_rope_dim_offsets = offs_rope.expandDims(1).mul(@as(i32, @intCast(config.stride_k_cache_3)));
        const k_rope_dim_ptrs = k.broadcastTo(key_block_ptrs, &.{ ROPE_RANK, TILE_SIZE })
            .addPtr(k.broadcastTo(k_rope_dim_offsets, &.{ ROPE_RANK, TILE_SIZE }));
        const slot_t = slot.expandDims(0);
        const key_slot_offsets = slot_t.to(.i64).mul(stride_k_cache_1);
        const K_rope = k.loadOpts(k_rope_dim_ptrs.addPtr(k.broadcastTo(key_slot_offsets, &.{ ROPE_RANK, TILE_SIZE })), .{
            .mask = valid_t.expandDims(0),
            .other = k.zeros(&.{ ROPE_RANK, TILE_SIZE }, config.kv_dtype),
            .cache_modifier = if (config.all_decode) .cg else .none,
        });
        S = S.add(scale.mul(k.dot(Q_rope, K_rope, k.zeros(&.{ BLOCK_M, TILE_SIZE }, .f32))));

        const k_lora_dim_offsets = offs_lora.expandDims(1).mul(@as(i32, @intCast(config.stride_k_cache_3)));
        const k_lora_dim_ptrs = k.broadcastTo(key_block_ptrs, &.{ KV_LORA_RANK, TILE_SIZE })
            .addPtr(k.broadcastTo(k_lora_dim_offsets, &.{ KV_LORA_RANK, TILE_SIZE }));
        const k_lora_mask = valid_t.expandDims(0).bitAnd(offs_lora.lt(@as(i32, @intCast(QK_LORA_RANK))).expandDims(1));
        const K_lora = k.loadOpts(k_lora_dim_ptrs.addPtr(k.broadcastTo(key_slot_offsets, &.{ KV_LORA_RANK, TILE_SIZE })), .{
            .mask = k_lora_mask,
            .other = k.zeros(&.{ KV_LORA_RANK, TILE_SIZE }, config.kv_dtype),
            .cache_modifier = if (config.all_decode) .cg else .none,
        });

        S = S.add(scale.mul(k.dot(Q_lora, K_lora, k.zeros(&.{ BLOCK_M, TILE_SIZE }, .f32))));

        const keep_mask = query_mask_1.expandDims(1)
            .bitAnd(query_mask_0.expandDims(1))
            .bitAnd(valid_t.expandDims(0));
        S = k.where(keep_mask, S, k.full(&.{ BLOCK_M, TILE_SIZE }, -std.math.inf(f32), .f32));

        var m_j = M.maximum(k.maxOpts(S, .{ .axis = 1 }));
        m_j = k.where(m_j.gt(-std.math.inf(f32)), m_j, k.full(&.{BLOCK_M}, 0.0, .f32));
        const P = k.exp(S.sub(m_j.expandDims(1)));
        const l_j = k.sumOpts(P, .{ .axis = 1 });
        const alpha = k.exp(M.sub(m_j));

        const acc_scaled = acc.mul(alpha.expandDims(1));
        const new_L = L.mul(alpha).add(l_j);

        const physical_block_idx_v = physical_block_idx.expandDims(1).to(.i64);
        const v_block_offsets = physical_block_idx_v.mul(stride_v_cache_0);
        const v_base_ptr = k.splat(value_cache_ptr, &.{ TILE_SIZE, 1 });
        const slot_v = slot.expandDims(1).to(.i64);
        const v_slot_offsets = slot_v.mul(stride_v_cache_1);
        const v_base_offsets = v_block_offsets.add(v_slot_offsets);
        const v_base_ptrs = v_base_ptr.addPtr(v_base_offsets);
        const v_lora_dim_offsets = offs_value.expandDims(0).mul(@as(i32, @intCast(config.stride_v_cache_3)));
        const V_lora = k.loadOpts(k.broadcastTo(v_base_ptrs, &.{ TILE_SIZE, VALUE_RANK }).addPtr(k.broadcastTo(v_lora_dim_offsets, &.{ TILE_SIZE, VALUE_RANK })), .{
            .mask = valid_t.expandDims(1),
            .other = k.zeros(&.{ TILE_SIZE, VALUE_RANK }, config.kv_dtype),
            .cache_modifier = if (config.all_decode) .cg else .none,
        });

        const new_acc = k.dot(P.to(config.kv_dtype), V_lora, acc_scaled);

        loop.yield(.{ m_j, new_L, new_acc });
    }

    const M = loop.results[0];
    var L = loop.results[1];
    var acc = loop.results[2];

    if (config.use_attn_sink) {
        const sink_mask = query_mask_0.bitAnd(query_mask_1);
        const sink_logits = k.loadOpts(attn_sink_ptr.addPtr(query_offset_1), .{
            .mask = sink_mask,
            .other = k.zeros(&.{BLOCK_M}, config.sink_dtype),
        }).to(.f32);
        const sink_score = k.where(
            sink_mask,
            sink_logits,
            k.full(&.{BLOCK_M}, -std.math.inf(f32), .f32),
        );
        const n_m = M.maximum(sink_score);
        const alpha = k.exp(M.sub(n_m));
        const sink_p = k.exp(sink_score.sub(n_m));

        acc = acc.mul(alpha.expandDims(1));
        L = L.mul(alpha).add(sink_p);
    }

    const one_over_L = k.full(&.{ BLOCK_M, 1 }, 1.0, .f32).div(L.expandDims(1));
    acc = acc.mul(k.broadcastTo(one_over_L, &.{ BLOCK_M, VALUE_RANK }));

    const output_offs_lora = query_offset_0.expandDims(1).mul(output_stride_0)
        .add(query_offset_1.expandDims(1).mul(output_stride_1))
        .add(offs_value.expandDims(0));
    k.storeOpts(
        output_ptr.addPtr(output_offs_lora),
        acc.to(config.o_dtype),
        .{ .mask = q_mask },
    );
}
