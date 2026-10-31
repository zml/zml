const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../../../../zml.zig");
const paged = @import("../../paged_attention.zig");
const cute = @import("../cute.zig");
const kernel = @import("attention.zig");

pub const Options = cute.Options;
pub const Parameters = cute.Parameters;

test {
    _ = kernel;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major == 10;
}

fn decodeSplits(work: i64, max_splits: i64, multiprocessors: i64, windowed: bool) i64 {
    // Short windows already expose enough independent heads for one wave.
    // Long contexts benefit from two resident CTAs per SM. Rounding down to
    // powers of two avoids a nearly empty second wave (e.g. 5 -> 4 splits).
    const target = multiprocessors * @as(i64, if (windowed) 1 else 2);
    const parallel = @divTrunc(target + work - 1, work);
    const limit: u64 = @intCast(@max(1, @min(32, @min(parallel, max_splits))));
    return @as(i64, 1) << @intCast(std.math.log2_int(u64, limit));
}

test "decode splits balance long contexts and short windows" {
    try std.testing.expectEqual(@as(i64, 4), decodeSplits(16 * 4, 32, 152, false));
    try std.testing.expectEqual(@as(i64, 2), decodeSplits(32 * 4, 32, 152, false));
    try std.testing.expectEqual(@as(i64, 2), decodeSplits(64 * 4, 32, 152, false));
    try std.testing.expectEqual(@as(i64, 1), decodeSplits(16 * 16, 5, 152, true));
    try std.testing.expectEqual(@as(i64, 1), decodeSplits(1, 1, 152, false));
}

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: paged.AttentionOptions) zml.Tensor {
    return zml.ops.manualComputation(Context.body, Context{ .q = q, .k = k, .v = v, .parameters = parameters, .opts = opts }, q.shape());
}

const Context = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    parameters: Parameters,
    opts: paged.AttentionOptions,

    fn body(self: Context, output_shape: zml.Shape) zml.Tensor {
        const q = self.q.transpose(.{ .b, .hkv, .hg, .hd });
        const k = self.k.transpose(.{ .page, .k_chunk, .hkv, .hd });
        const v = self.v.transpose(.{ .page, .k_chunk, .hkv, .hd });
        stdx.debug.assert(q.dim(.hkv) == k.dim(.hkv) and k.shape().eql(v.shape()) and k.dim(.hd) == q.dim(.hd) and q.dim(.hg) > 0, "CuTe paged attention requires aligned local Q/KV heads and matching caches", .{});
        const p = self.parameters;
        const decode = !p.options_.is_prefill;
        const cc = zml.platform.cuda.computeCapability(zml.Compiler.current().platform).?;
        var c: kernel.Config = .{
            .architecture = if (cc.minor >= 3) .sm103 else .sm100,
            .head_dim = q.dim(.hd),
            .tokens = q.dim(.b),
            .batch = p.block_table.dim(.b),
            .kv_heads = k.dim(.hkv),
            .group = q.dim(.hg),
            .pages_per_seq = p.block_table.dim(.p),
            .page_size = k.dim(.k_chunk),
            .max_query = if (decode) 1 else @min(q.dim(.b), @as(i64, @intCast(p.options_.max_seqlen_q))),
            .dtype = zml.kernel.cute.from(q.dtype()),
            .decode = decode,
            .causal = self.opts.mask.isCausal(),
            .window = self.opts.mask.slidingWindow() orelse 0,
            .scale = self.opts.scale orelse 1 / @sqrt(@as(f32, @floatFromInt(q.dim(.hd)))),
            .sink = self.opts.sink != null,
        };
        // Decode metadata reserves the server's maximum batch, including empty
        // rows. A compiled token-count variant bounds the active sequences.
        const query_tiles = if (decode) @min(c.batch, c.tokens) * c.queryTiles() else c.batch + @divTrunc(c.tokens * c.group + c.rows() - 1, c.rows());
        const work = query_tiles * c.kv_heads;
        const bound = if (c.window > 0 and c.causal) @min(c.window + @divTrunc(c.rows() + c.group - 1, c.group), c.pages_per_seq * c.page_size) else c.pages_per_seq * c.page_size;
        const multiprocessors: i64 = @intCast(cute.multiprocessorCount(zml.Compiler.current().platform));
        const max_splits = @divTrunc(bound + 2 * c.keys() - 1, 2 * c.keys());
        c.splits = if (decode)
            decodeSplits(work, max_splits, multiprocessors, c.window > 0 and c.causal)
        else
            @max(1, @min(32, @min(@divTrunc(multiprocessors * 2 + work - 1, work), max_splits)));
        // An unused scalar stays a constant buffer; broadcasting a dummy for
        // every head would add a helper kernel to each attention invocation.
        const sink = if (self.opts.sink) |s| s.convert(.f32).reshape(.{c.kv_heads * c.group}) else zml.Tensor.scalar(0, .f32).reshape(.{1});
        const out = kernel.Attention.call(.{ .q = q, .k = k, .v = v, .table = p.block_table, .lengths = p.seq_lens, .starts = p.query_start_len, .sink = sink }, .{
            .out = if (c.splits > 1) zml.Shape.init(.{ c.splits, c.tokens, c.kv_heads * c.group, c.head_dim + 2 }, .f32) else q.shape(),
        }, .{ .cfg = c, .grid = .{ @intCast(c.batch * c.queryTiles()), @intCast(c.kv_heads), @intCast(c.splits) }, .block = kernel.block }).out;
        const result = if (c.splits > 1) kernel.Merge.call(.{ .partials = out, .lengths = p.seq_lens, .starts = p.query_start_len, .sink = sink }, .{ .out = q.shape() }, .{
            .cfg = c,
            .grid = .{ @intCast(c.tokens), @intCast(c.kv_heads * c.group), 1 },
            .block = .{ 128, 1, 1 },
        }).out else out;
        return result.transpose(output_shape);
    }
};

const ArchitectureCheck = struct {
    fn forward(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, c: kernel.Config) zml.Tensor {
        const table_values: [9]i32 = .{ 4, 2, 8, 1, 6, 0, 7, 3, 5 };
        const table = zml.Tensor.constantTensor(.init(.{ 1, 9 }, .i32), std.mem.asBytes(&table_values));
        const lengths = zml.Tensor.constantTensor(.init(.{1}, .i32), std.mem.asBytes(&[_]i32{129}));
        const starts = zml.Tensor.constantTensor(.init(.{2}, .i32), std.mem.asBytes(&[_]i32{ 0, 3 }));
        const sink = zml.Tensor.zeroes(.init(.{3}, .f32));
        const partials = kernel.Attention.call(.{ .q = q, .k = k, .v = v, .table = table, .lengths = lengths, .starts = starts, .sink = sink }, .{
            .out = .init(.{ 2, 3, 3, c.head_dim + 2 }, .f32),
        }, .{ .cfg = c, .grid = .{ 1, 1, 2 }, .block = kernel.block }).out;
        return kernel.Merge.call(.{ .partials = partials, .lengths = lengths, .starts = starts, .sink = sink }, .{ .out = q.shape() }, .{
            .cfg = c,
            .grid = .{ 3, 3, 1 },
            .block = .{ 128, 1, 1 },
        }).out;
    }
};

test "SM10x GB200 and GB300 instruction paths match CPU reference" {
    const platform = zml.testing.env();
    if (!isAvailable(platform)) return error.SkipZigTest;
    const cc = zml.platform.cuda.computeCapability(platform).?;
    const a = std.testing.allocator;
    const io = std.testing.io;
    const BF16 = zml.floats.BFloat16;
    const table: [9]usize = .{ 4, 2, 8, 1, 6, 0, 7, 3, 5 };
    for ([_]i64{ 128, 512 }) |head_dim| {
        const d: usize = @intCast(head_dim);
        const q_host = try a.alloc(BF16, 9 * d);
        defer a.free(q_host);
        const k_host = try a.alloc(BF16, 144 * d);
        defer a.free(k_host);
        const v_host = try a.alloc(BF16, 144 * d);
        defer a.free(v_host);
        for (q_host, 0..) |*value, i| value.* = .fromF32(@as(f32, @floatFromInt(i % 13)) / 16 - 0.375);
        @memset(k_host, .fromF32(std.math.nan(f32)));
        @memset(v_host, .fromF32(std.math.nan(f32)));
        for (0..129) |key| for (0..d) |feature| {
            const offset = (table[key / 16] * 16 + key % 16) * d + feature;
            k_host[offset] = .fromF32(@as(f32, @floatFromInt((key * 7 + feature) % 17)) / 16 - 0.5);
            v_host[offset] = .fromF32(@as(f32, @floatFromInt((key + feature * 3) % 23)) / 16 - 0.625);
        };
        const q: zml.Tensor = .init(.{ .b = 3, .hkv = 1, .hg = 3, .hd = d }, .bf16);
        const k: zml.Tensor = .init(.{ .page = 9, .k_chunk = 16, .hkv = 1, .hd = d }, .bf16);
        const v: zml.Tensor = .fromShape(k.shape());
        var qb = try zml.Buffer.fromBytes(io, platform, q.shape(), std.mem.sliceAsBytes(q_host));
        defer qb.deinit();
        var kb = try zml.Buffer.fromBytes(io, platform, k.shape(), std.mem.sliceAsBytes(k_host));
        defer kb.deinit();
        var vb = try zml.Buffer.fromBytes(io, platform, k.shape(), std.mem.sliceAsBytes(v_host));
        defer vb.deinit();
        for ([_]kernel.Architecture{ .sm100, .sm103 }) |architecture| {
            if (architecture == .sm103 and cc.minor < 3) continue;
            const c: kernel.Config = .{ .architecture = architecture, .head_dim = head_dim, .tokens = 3, .batch = 1, .kv_heads = 1, .group = 3, .pages_per_seq = 9, .page_size = 16, .max_query = 3, .dtype = .bf16, .decode = false, .causal = true, .window = 0, .scale = 0.125, .sink = true, .splits = 2 };
            const exe = try platform.compileFn(a, io, ArchitectureCheck.forward, .{ q, k, v, c }, .{});
            defer exe.deinit();
            var output = try exe.eval(a, io, .{ qb, kb, vb });
            defer output.deinit();
            var host = try output.toSliceAlloc(a, io);
            defer host.free(a);
            const actual = host.items(BF16);
            for (0..9) |row| {
                const keys = 127 + row / 3;
                var scores: [129]f32 = undefined;
                var maximum: f32 = 0; // Sink score, with zero value.
                for (0..keys) |key| {
                    const offset = (table[key / 16] * 16 + key % 16) * d;
                    var score: f32 = 0;
                    for (0..d) |feature| score += q_host[row * d + feature].toF32() * k_host[offset + feature].toF32();
                    scores[key] = score * c.scale;
                    maximum = @max(maximum, scores[key]);
                }
                var denominator = @exp(-maximum);
                for (0..keys) |key| {
                    scores[key] = @exp(scores[key] - maximum);
                    denominator += scores[key];
                }
                for (0..d) |feature| {
                    var expected: f32 = 0;
                    for (0..keys) |key| expected += scores[key] * v_host[(table[key / 16] * 16 + key % 16) * d + feature].toF32();
                    expected /= denominator;
                    const got = actual[row * d + feature].toF32();
                    try std.testing.expect(std.math.isFinite(got));
                    try std.testing.expect(@abs(got - expected) <= 0.003 + 0.02 * @abs(expected));
                }
            }
        }
    }
}
