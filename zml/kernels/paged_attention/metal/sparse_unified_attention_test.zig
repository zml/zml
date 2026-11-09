const std = @import("std");
const zml = @import("../../../zml.zig");
const sparse = @import("sparse_unified_attention.zig");
const attention = @import("metal.zig");

test "Metal sparse attention emits verified modules" {
    for ([_]zml.kernel.metal.Type{ .half, .bfloat }) |dtype| {
        for ([_]i32{ 1, 33, 64, 128, 256, 512 }) |head_size| {
            for ([_]bool{ false, true }) |sinks| {
                inline for (.{ sparse.SparseAttention, sparse.SparseAttentionSplit }) |Producer| {
                    const source = try Producer.Kernel.emit(std.testing.allocator, .{
                        .dtype = dtype,
                        .head_size = head_size,
                        .num_heads = 4,
                        .num_queries_per_kv = 2,
                        .block_size = 4,
                        .indices_count = 7,
                        .max_blocks = 2,
                        .num_seqs = 4,
                        .use_sinks = sinks,
                        .num_segments = if (Producer == sparse.SparseAttentionSplit) 16 else 1,
                    });
                    std.testing.allocator.free(source);
                }
                const reduce = try sparse.SparseReduce.Kernel.emit(std.testing.allocator, .{
                    .dtype = dtype,
                    .head_size = head_size,
                    .num_heads = 4,
                    .num_segments = 16,
                    .launched_segments = 2,
                });
                std.testing.allocator.free(reduce);
            }
        }
    }
}

test "Metal sparse attention partitions selection slots" {
    try std.testing.expectEqual(sparse.Segments{ .span = 32, .launched = 2 }, sparse.Segments.init(33, 16));
    try std.testing.expectEqual(sparse.Segments{ .span = 192, .launched = 3 }, sparse.Segments.init(513, 3));
    try std.testing.expectEqual(@as(i32, 1), sparse.selectNumSegments(513, false));
    try std.testing.expectEqual(@as(i32, 1), sparse.selectNumSegments(256, true));
    try std.testing.expectEqual(@as(i32, 3), sparse.selectNumSegments(513, true));
    try std.testing.expectEqual(@as(i32, 8), sparse.selectNumSegments(8192, true));
}

const Case = struct {
    dim: usize,
    dtype: zml.DataType,
    sinks: bool,
    empty: bool = false,
    decode: bool = false,
    count: usize = 7,
    segments: ?i32 = null,
    holes: bool = false,
};
const Inputs = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    indices: zml.Tensor,
    table: zml.Tensor,
    lengths: zml.Tensor,
    starts: zml.Tensor,
    sinks: zml.Tensor,

    fn init(c: Case) Inputs {
        return .{
            .q = .init(.{ .b = 5, .hkv = 2, .hg = 2, .hd = c.dim }, .f32),
            .k = .init(.{ .page = 4, .k_chunk = 4, .hkv = 2, .hd = c.dim }, .f32),
            .v = .init(.{ .page = 4, .k_chunk = 4, .hkv = 2, .hd = c.dim }, .f32),
            .indices = .init(.{ .b = 5, .topk = c.count }, .i32),
            .table = .init(.{ .b = 4, .p = 2 }, .i32),
            .lengths = .init(.{ .b = 4 }, .i32),
            .starts = .init(.{ .b = 5 }, .i32),
            .sinks = .init(.{4}, .f32),
        };
    }
};

fn forward(i: Inputs, c: Case) zml.Tensor {
    if (c.segments) |segments| {
        const q = i.q.convert(c.dtype).merge(.{ .h = .{ .hkv, .hg } });
        return sparse.call(.{
            .q = q,
            .k = i.k.convert(c.dtype),
            .v = i.v.convert(c.dtype),
            .indices = i.indices,
            .block_tables = i.table,
            .seq_lens = i.lengths,
            .query_starts = i.starts,
            .sinks = i.sinks,
        }, q.shape(), .{
            .dtype = zml.kernel.metal.from(c.dtype),
            .head_size = @intCast(c.dim),
            .num_heads = 4,
            .num_queries_per_kv = 2,
            .block_size = 4,
            .indices_count = @intCast(c.count),
            .max_blocks = 2,
            .num_seqs = 4,
            .use_sinks = c.sinks,
            .num_segments = segments,
        }, 0.25).splitAxis(.h, .{ .hkv = 2, .hg = 2 }).convert(.f32);
    }
    return attention.pagedAttention(.{
        .block_table = i.table,
        .seq_lens = i.lengths,
        .query_start_len = i.starts,
        .options_ = .{ .batch_size = 4, .max_num_pages = 2, .max_seqlen_q = if (c.decode) 1 else 2, .is_prefill = !c.decode },
    }, i.q.convert(c.dtype), i.k.convert(c.dtype), i.v.convert(c.dtype), .{
        .mask = .{ .indices = i.indices },
        .sink = if (c.sinks) i.sinks else null,
        .scale = 0.25,
    }).convert(.f32);
}

test "Metal sparse attention matches CPU reference" {
    const platform = zml.testing.env();
    if (platform.target != .metal) return error.SkipZigTest;
    for ([_]Case{
        .{ .dim = 33, .dtype = .f16, .sinks = false },
        .{ .dim = 128, .dtype = .bf16, .sinks = true },
        .{ .dim = 512, .dtype = .bf16, .sinks = true, .decode = true, .count = 513 },
        .{ .dim = 33, .dtype = .f16, .sinks = false, .decode = true, .count = 257 },
        .{ .dim = 64, .dtype = .bf16, .sinks = true, .decode = true, .count = 513, .segments = 8, .holes = true },
        .{ .dim = 33, .dtype = .f16, .sinks = false, .decode = true, .count = 33, .segments = 16 },
        .{ .dim = 64, .dtype = .bf16, .sinks = false, .decode = true, .count = 513, .empty = true },
        .{ .dim = 64, .dtype = .f16, .sinks = true, .decode = true, .count = 513, .empty = true },
        .{ .dim = 64, .dtype = .bf16, .sinks = false, .empty = true },
        .{ .dim = 64, .dtype = .f16, .sinks = true, .empty = true },
    }) |c| try check(c, platform);
}

fn check(c: Case, platform: *const zml.Platform) !void {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();
    const spec = Inputs.init(c);
    const q = try alloc.alloc(f32, spec.q.count());
    const k = try alloc.alloc(f32, spec.k.count());
    const v = try alloc.alloc(f32, spec.v.count());
    const indices = try alloc.alloc(i32, spec.indices.count());
    for (q, 0..) |*x, j| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(j % 11)) - 5)) / 8;
    for (k, 0..) |*x, j| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(j % 13)) - 6)) / 8;
    for (v, 0..) |*x, j| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(j % 17)) - 8)) / 8;
    // Empty sequences, a repeated query start, prefill, and padded capacity.
    const starts: [5]i32 = if (c.decode) .{ 0, 1, 1, 2, 3 } else .{ 0, 2, 2, 3, 4 };
    const lengths = [_]i32{ 5, 0, 3, 0 };
    const table = [_]i32{ 2, 0, -1, -1, 1, -1, -1, -1 };
    const sinks = [_]f32{ -2, 0, 1, 3 };
    const pattern = [_]i32{ 4, -1, 0, 2, 0, 5, -7 };
    for (indices, 0..) |*index, j| index.* = pattern[j % c.count % pattern.len];
    if (c.holes) {
        @memset(indices, -1);
        for (0..5) |t| {
            indices[t * c.count + 200] = 2;
            indices[(t + 1) * c.count - 1] = 0;
        }
    }
    if (c.empty) {
        @memset(indices, -1);
        @memset(k, std.math.nan(f32));
        @memset(v, std.math.nan(f32));
    }
    var buffers: zml.Bufferized(Inputs) = undefined;
    var initialized: usize = 0;
    defer inline for (@typeInfo(Inputs).@"struct".field_names, 0..) |name, j| {
        if (j < initialized) @field(buffers, name).deinit();
    };
    const host = .{ .q = q, .k = k, .v = v, .indices = indices, .table = &table, .lengths = &lengths, .starts = &starts, .sinks = &sinks };
    inline for (@typeInfo(Inputs).@"struct".field_names) |name| {
        @field(buffers, name) = try zml.Buffer.fromBytes(io, platform, @field(spec, name).shape(), std.mem.sliceAsBytes(@field(host, name)));
        initialized += 1;
    }
    const exe = try platform.compileFn(a, io, forward, .{ spec, c }, .{});
    defer exe.deinit();
    var runner = try exe.runner(a);
    defer runner.deinit(a);
    var output: zml.Buffer = undefined;
    runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
    defer output.deinit();
    var result = try output.toSliceAlloc(a, io);
    defer result.free(a);
    const actual = result.items(f32);
    if (c.decode and (c.segments orelse sparse.selectNumSegments(@intCast(c.count), true)) > 1) {
        var direct_case = c;
        direct_case.segments = 1;
        const direct_exe = try platform.compileFn(a, io, forward, .{ spec, direct_case }, .{});
        defer direct_exe.deinit();
        var direct_runner = try direct_exe.runner(a);
        defer direct_runner.deinit(a);
        var direct_output: zml.Buffer = undefined;
        direct_runner.run(io, .{buffers}, .{&direct_output}, .{ .wait = true });
        defer direct_output.deinit();
        try zml.testing.expectClose(io, output, direct_output, .{ .absolute_tolerance = 0.004, .relative_tolerance = 0.004, .minimum_close_fraction = 1 });
    }
    const scores = try alloc.alloc(f32, c.count);
    for (0..5) |t| {
        const active = t < starts[4];
        var seq: usize = 0;
        while (seq < 3 and starts[seq + 1] <= t) : (seq += 1) {}
        for (0..4) |h| {
            var maximum: f32 = if (c.sinks) sinks[h] else -std.math.inf(f32);
            for (indices[t * c.count ..][0..c.count], 0..) |pos, slot| {
                scores[slot] = -std.math.inf(f32);
                if (!active or pos < 0 or pos >= lengths[seq]) continue;
                const p: usize = @intCast(pos);
                const page: usize = @intCast(table[seq * 2 + p / 4]);
                const off = ((page * 4 + p % 4) * 2 + h / 2) * c.dim;
                var dot: f32 = 0;
                for (0..c.dim) |d| dot += q[(t * 4 + h) * c.dim + d] * k[off + d];
                scores[slot] = dot * 0.25;
                maximum = @max(maximum, scores[slot]);
            }
            if (!std.math.isFinite(maximum)) maximum = 0;
            var sum: f32 = if (c.sinks and active) @exp(sinks[h] - maximum) else 0;
            for (scores) |*score| {
                score.* = @exp(score.* - maximum);
                sum += score.*;
            }
            for (0..c.dim) |d| {
                var numerator: f32 = 0;
                for (indices[t * c.count ..][0..c.count], scores) |pos, weight| {
                    if (weight == 0) continue;
                    const p: usize = @intCast(pos);
                    const page: usize = @intCast(table[seq * 2 + p / 4]);
                    numerator += weight * v[((page * 4 + p % 4) * 2 + h / 2) * c.dim + d];
                }
                const expected = numerator / (if (sum > 0) sum else 1);
                try std.testing.expectApproxEqAbs(expected, actual[(t * 4 + h) * c.dim + d], @as(f32, if (c.dtype == .bf16) 0.004 else 0.0005));
            }
        }
    }
}
