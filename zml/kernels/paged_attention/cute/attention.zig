const std = @import("std");

const zml = @import("../../../zml.zig");
const cute = zml.kernel.cute;
const V = cute.Value;

pub const Config = struct {
    head_dim: i64,
    tokens: i64,
    batch: i64,
    kv_heads: i64,
    group: i64,
    pages: i64,
    pages_per_seq: i64,
    page_size: i64,
    dtype: cute.DType,
    causal: bool,
    window: i64,
    scale: f32,
    sink: bool,
    indexed: bool,
    topk: i64,
};

pub const Kernel = cute.Kernel(Config, .{
    .name = "paged_attention_cute_generic",
    .inputs = &.{ "q", "k", "v", "table", "lengths", "starts", "sink", "indices" },
    .outputs = &.{"out"},
    .run = run,
});

fn run(b: *cute.Builder, c: Config) cute.FinishError!void {
    switch (c.head_dim) {
        32 => try runDim(b, c, 1),
        64 => try runDim(b, c, 2),
        128 => try runDim(b, c, 4),
        256 => try runDim(b, c, 8),
        512 => try runDim(b, c, 16),
        else => unreachable,
    }
}

fn runDim(b: *cute.Builder, c: Config, comptime chunks: usize) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .q = .{ .tensor = .{ .dtype = c.dtype, .shape = &.{ c.tokens, c.kv_heads * c.group, c.head_dim } } },
        .k = .{ .tensor = .{ .dtype = c.dtype, .shape = &.{ c.pages, c.page_size, c.kv_heads, c.head_dim } } },
        .v = .{ .tensor = .{ .dtype = c.dtype, .shape = &.{ c.pages, c.page_size, c.kv_heads, c.head_dim } } },
        .table = .{ .tensor = .{ .dtype = .i32, .shape = &.{ c.batch, c.pages_per_seq } } },
        .lengths = .{ .tensor = .{ .dtype = .i32, .shape = &.{c.batch} } },
        .starts = .{ .tensor = .{ .dtype = .i32, .shape = &.{c.batch + 1} } },
        .sink = .{ .tensor = .{ .dtype = .f32, .shape = &.{c.kv_heads * c.group} } },
        .indices = .{ .tensor = .{ .dtype = .i32, .shape = &.{ c.tokens, c.topk } } },
        .out = .{ .tensor = .{ .dtype = c.dtype, .shape = &.{ c.tokens, c.kv_heads * c.group, c.head_dim } } },
    });
    const token = b.blockIdx().x;
    const head = b.blockIdx().y;
    const lane = b.threadIdx().x;
    const zero = b.constant(.f32, 0);
    // One full warp owns a query/head; all control flow is warp-uniform.
    // Also define output for unused capacity at the end of the packed queries.
    for (0..chunks) |i| a.out.set(.{ token, head, lane.add(i * 32) }, zero.to(c.dtype));
    var active = b.openIf(token.lt(a.starts.get(.{c.batch})));
    {
        var find_seq = b.openFor(0, c.batch, 1, .{b.constant(.i32, 0)});
        {
            const owns = token.ge(a.starts.get(.{find_seq.iv})).bitAnd(token.lt(a.starts.get(.{find_seq.iv.add(1)})));
            find_seq.yield(.{b.select(owns, find_seq.iv, find_seq.carried[0])});
        }
        const seq = find_seq.results[0];
        const start = a.starts.get(.{seq});
        const qlen = a.starts.get(.{seq.add(1)}).sub(start);
        const len = a.lengths.get(.{seq});
        const position = len.sub(qlen).add(token.sub(start));
        const begin = if (c.window > 0) position.sub(c.window - 1).maximum(0) else b.constant(.i32, 0);
        const end = if (c.causal) position.add(1).minimum(len) else len;
        var query: [chunks]V = undefined;
        for (&query, 0..) |*q, i| q.* = a.q.get(.{ token, head, lane.add(i * 32) }).to(.f32);

        // State is running max, normalizer, then each lane's output features.
        // A sink is an initial softmax entry with zero value.
        var initial: [chunks + 2]V = @splat(zero);
        initial[0] = if (c.sink) a.sink.get(.{head}).mul(std.math.log2e) else b.constant(.f32, -std.math.inf(f32));
        initial[1] = if (c.sink) b.constant(.f32, 1) else zero;
        var keys = b.openFor(if (c.indexed) b.constant(.i32, 0) else begin, if (c.indexed) b.constant(.i32, c.topk) else end, 1, initial);
        {
            const key = if (c.indexed) a.indices.get(.{ token, keys.iv }) else keys.iv;
            const valid = key.ge(0).bitAnd(key.lt(len));
            // Invalid indexed entries must not load the page table, K, or V. Duplicates
            // deliberately contribute once per occurrence, as independent softmax entries.
            var entry = b.openIfElse(valid, @as([chunks + 2]V, @splat(zero)));
            {
                const page = a.table.get(.{ seq, key.div(c.page_size) });
                const offset = key.rem(c.page_size);
                const kv_head = head.div(c.group);
                var score = zero;
                for (query, 0..) |q, i| score = score.add(q.mul(a.k.get(.{ page, offset, kv_head, lane.add(i * 32) }).to(.f32)));
                inline for (.{ 16, 8, 4, 2, 1 }) |step| score = score.add(b.shuffleXor(score, step));
                score = score.mul(c.scale * std.math.log2e);
                const maximum = keys.carried[0].maximum(score);
                const alpha = b.exp2Approx(keys.carried[0].sub(maximum));
                const weight = b.exp2Approx(score.sub(maximum));
                var next: [chunks + 2]V = undefined;
                next[0] = maximum;
                next[1] = keys.carried[1].mul(alpha).add(weight);
                for (0..chunks) |i| next[i + 2] = keys.carried[i + 2].mul(alpha).add(weight.mul(a.v.get(.{ page, offset, kv_head, lane.add(i * 32) }).to(.f32)));
                entry.yieldThen(next);
            }
            {
                entry.yieldElse(keys.carried);
            }
            keys.yield(entry.results);
        }
        const norm = b.select(keys.results[1].gt(0), b.rcpApprox(keys.results[1]), zero);
        for (0..chunks) |i| a.out.set(.{ token, head, lane.add(i * 32) }, keys.results[i + 2].mul(norm).to(c.dtype));
        active.close();
    }
}
