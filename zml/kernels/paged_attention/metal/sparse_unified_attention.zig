//! Sparse paged attention over selection slots, including repeated positions.
const std = @import("std");
const metal = @import("../../../kernel.zig").metal;
const Builder = metal.Builder;
const zml = @import("../../../zml.zig");

const empty_max = -std.math.floatMax(f32) / 2;

/// Partition selection slots, rather than logical cache positions.
pub const Segments = struct {
    span: i32,
    launched: i32,

    pub fn init(count: i32, capacity: i32) Segments {
        std.debug.assert(count > 0 and capacity > 0);
        const tiles = std.math.divCeil(i32, count, 32) catch unreachable;
        const span = (std.math.divCeil(i32, tiles, capacity) catch unreachable) * 32;
        return .{ .span = span, .launched = std.math.divCeil(i32, count, span) catch unreachable };
    }
};

/// Keep prefill and short selections direct; split long decode selections.
pub fn selectNumSegments(count: i32, decode: bool) i32 {
    std.debug.assert(count > 0);
    return if (decode) @min(8, std.math.divCeil(i32, count, 256) catch unreachable) else 1;
}

pub const SparseAttentionConfig = struct {
    dtype: metal.Type,
    head_size: i32,
    num_heads: i32,
    num_queries_per_kv: i32,
    block_size: i32,
    indices_count: i32,
    max_blocks: i32,
    num_seqs: i32,
    use_sinks: bool,
    num_segments: i32 = 1,
};

pub const SparseAttention = Attention(false);
pub const SparseAttentionSplit = Attention(true);

fn Attention(comptime split_k: bool) type {
    return struct {
        pub const Config = SparseAttentionConfig;

        pub const Kernel = metal.Kernel(Config, .{
            .name = if (split_k) "sparse_unified_attention_split" else "sparse_unified_attention",
            .inputs = &.{ "q", "k", "v", "indices", "block_tables", "seq_lens", "query_starts", "sinks" },
            .outputs = &.{"out"},
            .constants = &.{"scale"},
            .run = run,
        });

        fn run(b: *Builder, cfg: Config) metal.FinishError!void {
            const input: metal.ArgSpec = .{ .buffer = .{ .type = cfg.dtype, .const_ = true } };
            const int_input: metal.ArgSpec = .{ .buffer = .{ .type = .int, .const_ = true } };
            const a = try b.declareArgs(.{
                .scale = .{ .constant = .float },
                .q = input,
                .k = input,
                .v = input,
                .indices = int_input,
                .block_tables = int_input,
                .seq_lens = int_input,
                .query_starts = int_input,
                .sinks = .{ .buffer = .{ .type = .float, .const_ = true } },
                .out = .{ .buffer = .{ .type = if (split_k) .float else cfg.dtype } },
                .group = .{ .threadgroup_position_in_grid = .uint3 },
                .lane = .{ .thread_index_in_simdgroup = .ushort },
            });
            const query = a.group.x().to(.int);
            const head = a.group.y().to(.int);
            const lane = a.lane.to(.int);
            std.debug.assert(split_k or cfg.num_segments == 1);
            const partition = Segments.init(cfg.indices_count, cfg.num_segments);
            const slot = a.group.z().to(.int);
            // Launch the partial tail first, without launching unused capacity.
            const segment = if (split_k) b.ternary(slot.eq(0), partition.launched - 1, slot.sub(1)) else b.constant(.int, 0);
            const row_idx = query.to(.long).mul(cfg.num_heads).add(head);
            const row = row_idx.mul(cfg.head_size);
            // Split partials are FP32: [query, head, segment, head_size + 2].
            // The last two elements hold the running maximum and exponential sum.
            const partial_row = row_idx.mul(cfg.num_segments).add(segment).mul(cfg.head_size + 2);
            const chunks = @divTrunc(cfg.head_size + 31, 32);
            const acc = b.zeroedArray(.float, @intCast(chunks));

            // Padded query capacity is not part of any sequence.
            var padding = b.openIf(query.ge(a.query_starts.load(cfg.num_seqs)));
            {
                if (!split_k) {
                    var dims = b.openFor(lane, cfg.head_size, 32);
                    {
                        a.out.store(row.add(dims.iv), b.constant(cfg.dtype, 0));
                        dims.close();
                    }
                }
                b.return_();
                padding.close();
            }

            const left = b.local(.int, 0);
            const right = b.local(.int, cfg.num_seqs);
            var search = b.openLoop(.{});
            search.body(left.lt(right));
            {
                const mid = left.add(right).div(2);
                var half = b.openIfElse(a.query_starts.load(mid).le(query), .{});
                {
                    left.assign(mid.add(1));
                    half.elseBranch();
                }
                {
                    right.assign(mid);
                    half.close();
                }
                search.close();
            }
            const seq = left.sub(1);
            const seq_len = a.seq_lens.load(seq);
            const kv_head = head.div(cfg.num_queries_per_kv);
            const kv_heads = @divExact(cfg.num_heads, cfg.num_queries_per_kv);
            const maximum = b.local(.float, empty_max);
            const sum = b.local(.float, 0.0);
            if (cfg.use_sinks) {
                // A sink contributes denominator mass once, in logical segment 0.
                var sink = b.openIf(segment.eq(0));
                maximum.assign(a.sinks.load(head));
                sum.assign(1.0);
                sink.close();
            }
            const q_reg = b.zeroedArray(.float, @intCast(chunks));
            var load_q = b.openFor(0, chunks, 1);
            {
                const dim = lane.add(load_q.iv.mul(32));
                var valid = b.openIf(dim.lt(cfg.head_size));
                {
                    q_reg.store(load_q.iv, a.q.load(row.add(dim)).to(.float));
                    valid.close();
                }
                load_q.close();
            }
            const start = if (split_k) segment.mul(partition.span) else b.constant(.int, 0);
            const end = if (split_k) b.min(start.add(partition.span), cfg.indices_count) else b.constant(.int, cfg.indices_count);
            var slots = b.openFor(start, end, 1);
            {
                const position = a.indices.load(query.to(.long).mul(cfg.indices_count).add(slots.iv));
                // The branch is uniform across the SIMD group. Invalid positions
                // never access the page table or caches (which may contain NaNs).
                var valid = b.openIf(position.ge(0).bitAnd(position.lt(seq_len)));
                {
                    const page = a.block_tables.load(seq.to(.long).mul(cfg.max_blocks).add(position.div(cfg.block_size))).to(.long);
                    const offset = page.mul(cfg.block_size).add(position.rem(cfg.block_size)).mul(kv_heads).add(kv_head).mul(cfg.head_size);
                    const dot = b.local(.float, 0.0);
                    var dims = b.openFor(0, chunks, 1);
                    {
                        const dim = lane.add(dims.iv.mul(32));
                        var present = b.openIf(dim.lt(cfg.head_size));
                        {
                            dot.assign(dot.add(q_reg.load(dims.iv).mul(a.k.load(offset.add(dim)).to(.float))));
                            present.close();
                        }
                        dims.close();
                    }
                    const score = b.simdSum(dot).mul(a.scale);
                    const next_max = b.max(maximum.rvalue(), score);
                    const alpha = b.ternary(sum.eq(0.0), 0.0, b.exp(maximum.sub(next_max)));
                    const weight = b.exp(score.sub(next_max));
                    var update = b.openFor(0, chunks, 1);
                    {
                        const dim = lane.add(update.iv.mul(32));
                        var present = b.openIf(dim.lt(cfg.head_size));
                        {
                            acc.store(update.iv, acc.load(update.iv).mul(alpha).add(weight.mul(a.v.load(offset.add(dim)).to(.float))));
                            present.close();
                        }
                        update.close();
                    }
                    sum.assign(sum.mul(alpha).add(weight));
                    maximum.assign(next_max);
                    valid.close();
                }
                slots.close();
            }
            if (split_k) {
                var stats = b.openIf(lane.eq(0));
                a.out.store(partial_row.add(cfg.head_size), maximum);
                a.out.store(partial_row.add(cfg.head_size + 1), sum);
                stats.close();
            }
            const denominator = b.ternary(sum.gt(0.0), sum, 1.0);
            var store = b.openFor(0, chunks, 1);
            {
                const dim = lane.add(store.iv.mul(32));
                var present = b.openIf(dim.lt(cfg.head_size));
                {
                    if (split_k) {
                        a.out.store(partial_row.add(dim), acc.load(store.iv));
                    } else {
                        a.out.store(row.add(dim), acc.load(store.iv).div(denominator).to(cfg.dtype));
                    }
                    present.close();
                }
                store.close();
            }
        }
    };
}

/// Merge only launched, nonempty partials using their softmax statistics.
pub const SparseReduce = struct {
    pub const Config = struct {
        dtype: metal.Type,
        head_size: i32,
        num_heads: i32,
        num_segments: i32,
        launched_segments: i32,
    };
    pub const Kernel = metal.Kernel(Config, .{
        .name = "sparse_attention_reduce",
        .inputs = &.{ "partials", "query_starts" },
        .outputs = &.{"out"},
        .constants = &.{"num_seqs"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Config) metal.FinishError!void {
        const a = try b.declareArgs(.{
            .partials = .{ .buffer = .{ .type = .float, .const_ = true } },
            .query_starts = .{ .buffer = .{ .type = .int, .const_ = true } },
            .out = .{ .buffer = .{ .type = cfg.dtype } },
            .num_seqs = .{ .constant = .int },
            .group = .{ .threadgroup_position_in_grid = .uint3 },
            .lane = .{ .thread_index_in_simdgroup = .ushort },
        });
        const query = a.group.x().to(.int);
        const row = query.to(.long).mul(cfg.num_heads).add(a.group.y());
        const lane = a.lane.to(.int);
        var padding = b.openIf(query.ge(a.query_starts.load(a.num_seqs)));
        {
            var dims = b.openFor(lane, cfg.head_size, 32);
            a.out.store(row.mul(cfg.head_size).add(dims.iv), b.constant(cfg.dtype, 0));
            dims.close();
            b.return_();
            padding.close();
        }
        const chunks = @divTrunc(cfg.head_size + 31, 32);
        const acc = b.zeroedArray(.float, @intCast(chunks));
        const maximum = b.local(.float, empty_max);
        const sum = b.local(.float, 0.0);
        var segments = b.openFor(0, cfg.launched_segments, 1);
        {
            const base = row.mul(cfg.num_segments).add(segments.iv).mul(cfg.head_size + 2);
            const mass = a.partials.load(base.add(cfg.head_size + 1));
            // Empty segments and unused scratch capacity must not affect the
            // maximum or read partial numerators (which may be uninitialized).
            var nonempty = b.openIf(mass.gt(0.0));
            {
                const m = a.partials.load(base.add(cfg.head_size));
                const next_max = b.max(maximum.rvalue(), m);
                const alpha = b.ternary(sum.eq(0.0), 0.0, b.exp(maximum.sub(next_max)));
                const beta = b.exp(m.sub(next_max));
                var dims = b.openFor(0, chunks, 1);
                {
                    const dim = lane.add(dims.iv.mul(32));
                    var valid = b.openIf(dim.lt(cfg.head_size));
                    acc.store(dims.iv, acc.load(dims.iv).mul(alpha).add(a.partials.load(base.add(dim)).mul(beta)));
                    valid.close();
                    dims.close();
                }
                sum.assign(sum.mul(alpha).add(mass.mul(beta)));
                maximum.assign(next_max);
                nonempty.close();
            }
            segments.close();
        }
        const denominator = b.ternary(sum.gt(0.0), sum, 1.0);
        var dims = b.openFor(0, chunks, 1);
        {
            const dim = lane.add(dims.iv.mul(32));
            var valid = b.openIf(dim.lt(cfg.head_size));
            a.out.store(row.mul(cfg.head_size).add(dim), acc.load(dims.iv).div(denominator).to(cfg.dtype));
            valid.close();
            dims.close();
        }
    }
};

/// Inputs and output use the canonical contiguous layouts of the Metal backend.
pub fn call(inputs: SparseAttention.Kernel.Inputs, output: zml.Shape, cfg: SparseAttentionConfig, scale: f32) zml.Tensor {
    const grid: [3]u32 = .{ @intCast(output.dim(0)), @intCast(cfg.num_heads), 1 };
    if (cfg.num_segments == 1) return SparseAttention.Kernel.call(inputs, .{ .out = output }, .{
        .cfg = cfg,
        .constants = .{ .scale = .of(scale) },
        .grid = grid,
        .threadgroup = .{ 32, 1, 1 },
    }).out;
    const partition = Segments.init(cfg.indices_count, cfg.num_segments);
    const partials = SparseAttentionSplit.Kernel.call(inputs, .{
        .out = zml.Shape.init(.{ output.dim(0), cfg.num_heads, cfg.num_segments, cfg.head_size + 2 }, .f32),
    }, .{
        .cfg = cfg,
        .constants = .{ .scale = .of(scale) },
        .grid = .{ grid[0], grid[1], @intCast(partition.launched) },
        .threadgroup = .{ 32, 1, 1 },
    }).out;
    return SparseReduce.Kernel.call(.{ .partials = partials, .query_starts = inputs.query_starts }, .{ .out = output }, .{
        .cfg = .{
            .dtype = cfg.dtype,
            .head_size = cfg.head_size,
            .num_heads = cfg.num_heads,
            .num_segments = cfg.num_segments,
            .launched_segments = partition.launched,
        },
        .constants = .{ .num_seqs = .of(cfg.num_seqs) },
        .grid = grid,
        .threadgroup = .{ 32, 1, 1 },
    }).out;
}
