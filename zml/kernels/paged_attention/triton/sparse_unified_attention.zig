const std = @import("std");
const zml = @import("../../../zml.zig");
const tri = zml.kernel.triton;
const Builder = tri.Builder;
const Value = tri.Value;
const DType = tri.DType;

/// `log_2(e)` — pre-multiply with this to turn `exp(x)` into `exp2(x * RCP_LN2)`.
const RCP_LN2: f32 = 1.4426950408889634;

/// Partition selection slots, not logical KV positions. Scratch capacity may be
/// larger than the number of segments that have any selection slots to visit.
pub const SparseSegments = struct {
    tiles_per_segment: i64,
    span: i64,
    launched: i64,

    pub fn init(count: i64, capacity: i64, tile: i64) SparseSegments {
        std.debug.assert(count > 0 and capacity > 0 and tile > 0);
        const tiles = std.math.divCeil(i64, count, capacity * tile) catch unreachable;
        return .{
            .tiles_per_segment = tiles,
            .span = tiles * tile,
            .launched = std.math.divCeil(i64, count, tiles * tile) catch unreachable,
        };
    }
};

pub const SparseAttentionConfig = struct {
    dtype: DType,
    num_seqs: i32,
    num_query_heads: i64,
    num_queries_per_kv: i64,
    head_size: i64,
    head_size_padded: i64,
    block_m: i64,
    block_size: i64,
    tile_size: i64,
    indices_count: i64,
    num_segments: i64 = 1,
    decode_sequence_grid: bool = false,
    use_sinks: bool = false,
    scale: f32,
    block_table_stride: i64,
    query_strides: [3]i64,
    key_strides: [4]i64,
    value_strides: [4]i64,
    indices_strides: [2]i64,
};

fn findSeqIdx(
    k: *Builder,
    query_start_len_ptr: Value,
    target_idx: Value,
    num_seqs: Value,
    block_q: i64,
    use_q_block_mode: bool,
) Value {
    const left_init = k.liftAs(0, .i32);

    var w = k.openWhile(.{ left_init, num_seqs }, .{ .i32, .i32 });
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

        var i = k.openIfElse(mid_val.le(target_idx), .{ .i32, .i32 });
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

/// One program processes all selected keys for one query/KV-head group and write the normalized output.
pub const KernelSparseAttention2dPtr = KernelSparseAttentionPtr(false);

/// Also known as split-K in the litterature, this 3d variant uses multiple programs process different
/// portions of that same selection list. They write partial results, then a reduction kernels combines
/// them into one normalized output.
///
/// Mostly useful during decode when there are few queries and a long top-k list.
pub const KernelSparseAttention3dPtr = KernelSparseAttentionPtr(true);

/// One query and one GQA group per program. The direct specialization writes
/// normalized output; split decode writes FP32 partials for SparseReducePtr.
fn KernelSparseAttentionPtr(comptime split_k: bool) type {
    return struct {
        pub const Config = SparseAttentionConfig;
        pub const Kernel = tri.Kernel(Config, .{
            .name = if (split_k) "kernel_sparse_attention_3d_ptr" else "kernel_sparse_attention_2d_ptr",
            .inputs = &.{
                "query_ptr",        "key_ptr",      "value_ptr",           "indices_ptr",
                "block_tables_ptr", "seq_lens_ptr", "query_start_len_ptr", "sink_ptr",
            },
            .outputs = if (split_k) &.{ "output", "segm_max", "segm_expsum" } else &.{"output"},
            .run = run,
        });

        fn run(k: *Builder, cfg: Config) tri.FinishError!void {
            const a = if (split_k) try k.declareArgs(.{
                .query_ptr = .{ .ptr = cfg.dtype },
                .key_ptr = .{ .ptr = cfg.dtype },
                .value_ptr = .{ .ptr = cfg.dtype },
                .indices_ptr = .{ .ptr = .i32 },
                .block_tables_ptr = .{ .ptr = .i32 },
                .seq_lens_ptr = .{ .ptr = .i32 },
                .query_start_len_ptr = .{ .ptr = .i32 },
                .sink_ptr = .{ .ptr = .f32 },
                .output_ptr = .{ .ptr = .f32 },
                .segm_max_ptr = .{ .ptr = .f32 },
                .segm_expsum_ptr = .{ .ptr = .f32 },
            }) else try k.declareArgs(.{
                .query_ptr = .{ .ptr = cfg.dtype },
                .key_ptr = .{ .ptr = cfg.dtype },
                .value_ptr = .{ .ptr = cfg.dtype },
                .indices_ptr = .{ .ptr = .i32 },
                .block_tables_ptr = .{ .ptr = .i32 },
                .seq_lens_ptr = .{ .ptr = .i32 },
                .query_start_len_ptr = .{ .ptr = .i32 },
                .sink_ptr = .{ .ptr = .f32 },
                .output_ptr = .{ .ptr = cfg.dtype },
            });

            std.debug.assert(cfg.block_m >= cfg.num_queries_per_kv);
            std.debug.assert(split_k or cfg.num_segments == 1);

            const partition = SparseSegments.init(cfg.indices_count, cfg.num_segments, cfg.tile_size);
            const program = k.programId(.x);
            const num_seqs = k.liftAs(cfg.num_seqs, .i32);
            if (!cfg.decode_sequence_grid) {
                const active_queries = k.load(a.query_start_len_ptr.addPtr(num_seqs));
                k.returnIf(program.ge(active_queries));
            }

            const seq = if (cfg.decode_sequence_grid) program else findSeqIdx(k, a.query_start_len_ptr, program, num_seqs, 1, false);
            const seq_len = k.load(a.seq_lens_ptr.addPtr(seq));
            const query_start = k.load(a.query_start_len_ptr.addPtr(seq));
            const query_end = k.load(a.query_start_len_ptr.addPtr(seq).addPtr(1));
            if (cfg.decode_sequence_grid) k.returnIf(query_start.eq(query_end));
            const query = if (cfg.decode_sequence_grid) query_start else program;
            const kv_head = if (cfg.num_query_heads == cfg.num_queries_per_kv) k.liftAs(0, .i32) else k.programId(.y);

            // Launch the partial tail first, without launching unused capacity.
            const slot = if (split_k) k.programId(.z) else k.liftAs(0, .i32);
            const segment = if (split_k) k.where(slot.eq(0), k.liftAs(@as(i32, @intCast(partition.launched - 1)), .i32), slot.sub(1)) else slot;
            const segment_start = segment.mul(@as(i32, @intCast(partition.span)));
            const heads = k.arange(0, cfg.block_m, .i32);
            const head = kv_head.mul(@as(i32, @intCast(cfg.num_queries_per_kv))).add(heads);
            const head_mask = heads.lt(@as(i32, @intCast(cfg.num_queries_per_kv)));
            const dims = k.arange(0, cfg.head_size_padded, .i32);
            const dim_mask = dims.lt(@as(i32, @intCast(cfg.head_size)));
            const indices_row = a.indices_ptr.addPtr(query.to(.i64).mul(cfg.indices_strides[0]));

            const sink = if (cfg.use_sinks) k.loadOpts(a.sink_ptr.addPtr(head), .{
                .mask = head_mask.bitAnd(segment.eq(0)),
                .other = k.full(&.{cfg.block_m}, -std.math.inf(f32), .f32),
            }).mul(RCP_LN2) else k.full(&.{cfg.block_m}, -std.math.inf(f32), .f32);
            const has_sink = if (cfg.use_sinks) segment.eq(0).bitAnd(head_mask) else k.full(&.{cfg.block_m}, 0, .i1);
            const initial_max = k.where(has_sink, sink, k.full(&.{cfg.block_m}, -std.math.inf(f32), .f32));
            const initial_sum = has_sink.to(.f32);
            const initial_acc = k.zeros(&.{ cfg.block_m, cfg.head_size_padded }, .f32);

            if (partition.span <= 256) {
                const scan_size: i64 = @intCast(std.math.ceilPowerOfTwoAssert(u64, @intCast(partition.span)));
                const selected = segment_start.add(k.arange(0, scan_size, .i32));
                const scan_mask = selected.lt(cfg.indices_count).bitAnd(selected.lt(segment_start.add(@as(i32, @intCast(partition.span)))));
                const positions = k.loadOpts(indices_row.addPtr(selected.to(.i64).mul(cfg.indices_strides[1])), .{
                    .mask = scan_mask,
                    .other = k.full(&.{scan_size}, -1, .i32),
                });
                const empty = k.sum(scan_mask.bitAnd(positions.ge(0)).bitAnd(positions.lt(seq_len)).to(.i32)).eq(0);
                var branch = k.openIfElse(empty, .{});
                storeResult(k, cfg, a, query, head, head_mask, segment, dims, dim_mask, initial_max, initial_sum, initial_acc);
                branch.elseBranch();
                branch.close();
                k.returnIf(empty);
            }

            const q_offset = query.to(.i64).mul(cfg.query_strides[0])
                .add(head.to(.i64).mul(cfg.query_strides[1]).expandDims(1))
                .add(dims.to(.i64).mul(cfg.query_strides[2]).expandDims(0));
            const Q = k.loadOpts(a.query_ptr.addPtr(q_offset), .{
                .mask = k.mask2d(head_mask, dim_mask, cfg.block_m, cfg.head_size_padded),
                .other = initial_acc.to(cfg.dtype),
            });
            const table = a.block_tables_ptr.addPtr(seq.to(.i64).mul(cfg.block_table_stride));
            const offsets = k.arange(0, cfg.tile_size, .i32);
            var loop = k.openFor(k.liftAs(0, .i32), k.liftAs(@as(i32, @intCast(partition.tiles_per_segment)), .i32), 1, .{ initial_max, initial_sum, initial_acc });
            {
                const selected = segment_start.add(loop.iv.mul(@as(i32, @intCast(cfg.tile_size)))).add(offsets);
                const position = k.loadOpts(indices_row.addPtr(selected.to(.i64).mul(cfg.indices_strides[1])), .{
                    .mask = selected.lt(cfg.indices_count),
                    .other = k.full(&.{cfg.tile_size}, -1, .i32),
                });
                const valid = selected.lt(cfg.indices_count).bitAnd(position.ge(0)).bitAnd(position.lt(seq_len));
                // A bounded single-tile segment was already checked above.
                const nonempty = if (partition.tiles_per_segment == 1 and partition.span <= 256) k.liftAs(1, .i1) else k.sum(valid.to(.i32)).gt(0);
                var branch = k.openIfElse(nonempty, .{ k.tensorTy(&.{cfg.block_m}, .f32), k.tensorTy(&.{cfg.block_m}, .f32), k.tensorTy(&.{ cfg.block_m, cfg.head_size_padded }, .f32) });
                const safe_position = k.where(valid, position, k.zeros(&.{cfg.tile_size}, .i32));
                const page = k.loadOpts(table.addPtr(safe_position.div(@as(i32, @intCast(cfg.block_size)))), .{
                    .mask = valid,
                    .other = k.zeros(&.{cfg.tile_size}, .i32),
                }).to(.i64);
                const in_page = safe_position.rem(@as(i32, @intCast(cfg.block_size))).to(.i64);
                const key_offset = page.mul(cfg.key_strides[0]).add(in_page.mul(cfg.key_strides[1])).add(kv_head.to(.i64).mul(cfg.key_strides[2])).expandDims(0)
                    .add(dims.to(.i64).mul(cfg.key_strides[3]).expandDims(1));
                const value_offset = page.mul(cfg.value_strides[0]).add(in_page.mul(cfg.value_strides[1])).add(kv_head.to(.i64).mul(cfg.value_strides[2])).expandDims(1)
                    .add(dims.to(.i64).mul(cfg.value_strides[3]).expandDims(0));
                const K = k.loadOpts(a.key_ptr.addPtr(key_offset), .{
                    .mask = k.mask2d(dim_mask, valid, cfg.head_size_padded, cfg.tile_size),
                    .other = k.zeros(&.{ cfg.head_size_padded, cfg.tile_size }, cfg.dtype),
                });
                const V = k.loadOpts(a.value_ptr.addPtr(value_offset), .{
                    .mask = k.mask2d(valid, dim_mask, cfg.tile_size, cfg.head_size_padded),
                    .other = k.zeros(&.{ cfg.tile_size, cfg.head_size_padded }, cfg.dtype),
                });
                const scores = k.dot(Q, K, k.zeros(&.{ cfg.block_m, cfg.tile_size }, .f32)).mul(cfg.scale * RCP_LN2);
                const S = k.where(k.mask2d(head_mask, valid, cfg.block_m, cfg.tile_size), scores, k.full(&.{ cfg.block_m, cfg.tile_size }, -std.math.inf(f32), .f32));
                const max_score = loop.carried[0].maximum(k.maxOpts(S, .{ .axis = 1 }));
                const next_max = k.where(max_score.gt(-std.math.inf(f32)), max_score, k.zeros(&.{cfg.block_m}, .f32));
                const alpha = k.exp2(loop.carried[0].sub(next_max));
                const P = k.exp2(S.sub(next_max.expandDims(1)));
                const sum = loop.carried[1].mul(alpha).add(k.sumOpts(P, .{ .axis = 1 }));
                const acc = k.dot(P.to(cfg.dtype), V, loop.carried[2].mul(alpha.expandDims(1)));
                branch.yieldThen(.{ next_max, sum, acc });
                branch.yieldElse(.{ loop.carried[0], loop.carried[1], loop.carried[2] });
                loop.yield(.{ branch.results[0], branch.results[1], branch.results[2] });
            }

            storeResult(k, cfg, a, query, head, head_mask, segment, dims, dim_mask, loop.results[0], loop.results[1], loop.results[2]);
        }

        fn storeResult(k: *Builder, cfg: Config, a: anytype, query: Value, head: Value, head_mask: Value, segment: Value, dims: Value, dim_mask: Value, max_score: Value, sum: Value, acc: Value) void {
            if (split_k) {
                // Adjacent heads are written together by the producer.
                const stats_offset = query.to(.i64).mul(cfg.num_segments).add(segment.to(.i64)).mul(cfg.num_query_heads).add(head.to(.i64));
                k.storeOpts(a.segm_max_ptr.addPtr(stats_offset), max_score, .{ .mask = head_mask });
                k.storeOpts(a.segm_expsum_ptr.addPtr(stats_offset), sum, .{ .mask = head_mask });
                const offset = query.to(.i64).mul(cfg.num_query_heads).add(head.to(.i64)).mul(cfg.num_segments).add(segment.to(.i64)).mul(cfg.head_size_padded).expandDims(1).add(dims.to(.i64).expandDims(0));
                // Zero-mass segments need no partial output; the reducer masks them.
                k.storeOpts(a.output_ptr.addPtr(offset), acc, .{ .mask = k.mask2d(head_mask.bitAnd(sum.gt(0.0)), dim_mask, cfg.block_m, cfg.head_size_padded) });
            } else {
                const denominator = k.where(sum.gt(0.0), sum, k.full(&.{cfg.block_m}, 1.0, .f32));
                const offset = query.to(.i64).mul(cfg.query_strides[0]).add(head.to(.i64).mul(cfg.query_strides[1]).expandDims(1)).add(dims.to(.i64).mul(cfg.query_strides[2]).expandDims(0));
                k.storeOpts(a.output_ptr.addPtr(offset), acc.div(denominator.expandDims(1)).to(cfg.dtype), .{ .mask = k.mask2d(head_mask, dim_mask, cfg.block_m, cfg.head_size_padded) });
            }
        }
    };
}

pub const SparseReducePtr = struct {
    pub const Config = struct {
        dtype: DType,
        num_seqs: i32,
        num_query_heads: i64,
        head_size: i64,
        head_size_padded: i64,
        head_tile: i64,
        num_segments: i64,
        launched_segments: i64,
        output_strides: [3]i64,
    };
    pub const Kernel = tri.Kernel(Config, .{
        .name = "reduce_sparse_attention_ptr",
        .inputs = &.{ "segm_output_ptr", "segm_max_ptr", "segm_expsum_ptr", "query_start_len_ptr" },
        .outputs = &.{"output"},
        .run = run,
    });
    fn run(k: *Builder, cfg: Config) tri.FinishError!void {
        const a = try k.declareArgs(.{
            .segm_output_ptr = .{ .ptr = .f32 },
            .segm_max_ptr = .{ .ptr = .f32 },
            .segm_expsum_ptr = .{ .ptr = .f32 },
            .query_start_len_ptr = .{ .ptr = .i32 },
            .output_ptr = .{ .ptr = cfg.dtype },
        });
        const query = k.programId(.x);
        k.returnIf(query.ge(k.load(a.query_start_len_ptr.addPtr(cfg.num_seqs))));
        const head = k.programId(.y);
        const segments = k.arange(0, cfg.num_segments, .i32);
        const segment_mask = segments.lt(cfg.launched_segments);
        const dims = k.programId(.z).mul(@as(i32, @intCast(cfg.head_tile))).add(k.arange(0, cfg.head_tile, .i32));
        const dim_mask = dims.lt(cfg.head_size);
        const stats = query.to(.i64).mul(cfg.num_segments).add(segments.to(.i64)).mul(cfg.num_query_heads).add(head.to(.i64));
        const maxima = k.loadOpts(a.segm_max_ptr.addPtr(stats), .{ .mask = segment_mask, .other = k.full(&.{cfg.num_segments}, -std.math.inf(f32), .f32) });
        const sums = k.loadOpts(a.segm_expsum_ptr.addPtr(stats), .{ .mask = segment_mask, .other = k.zeros(&.{cfg.num_segments}, .f32) });
        const nonempty = segment_mask.bitAnd(sums.gt(0.0));
        const maximum = k.max(k.where(nonempty, maxima, k.full(&.{cfg.num_segments}, -std.math.inf(f32), .f32)));
        const safe_maximum = k.where(maximum.gt(-std.math.inf(f32)), maximum, k.liftAs(0.0, .f32));
        const rescale = k.exp2(maxima.sub(safe_maximum));
        const denominator = k.sum(sums.mul(rescale));
        const offset = query.to(.i64).mul(cfg.num_query_heads).add(head.to(.i64)).mul(cfg.num_segments).add(segments.to(.i64)).mul(cfg.head_size_padded).expandDims(1).add(dims.to(.i64).expandDims(0));
        const partial = k.loadOpts(a.segm_output_ptr.addPtr(offset), .{
            .mask = k.mask2d(nonempty, dim_mask, cfg.num_segments, cfg.head_tile),
            .other = k.zeros(&.{ cfg.num_segments, cfg.head_tile }, .f32),
        });
        const numerator = k.sumOpts(partial.mul(rescale.expandDims(1)), .{ .axis = 0 });
        const output = numerator.div(k.where(denominator.gt(0.0), denominator, k.liftAs(1.0, .f32)));
        const output_offset = query.to(.i64).mul(cfg.output_strides[0]).add(head.to(.i64).mul(cfg.output_strides[1])).add(dims.to(.i64).mul(cfg.output_strides[2]));
        k.storeOpts(a.output_ptr.addPtr(output_offset), output.to(cfg.dtype), .{ .mask = dim_mask });
    }
};
