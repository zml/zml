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
    return cc.major == 12;
}

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: paged.AttentionOptions) zml.Tensor {
    const q_ = q.transpose(.{ .b, .hkv, .hg, .hd });
    const k_ = k.transpose(.{ .page, .k_chunk, .hkv, .hd });
    const v_ = v.transpose(.{ .page, .k_chunk, .hkv, .hd });
    stdx.debug.assert(q_.dim(.hkv) == k_.dim(.hkv) and k_.shape().eql(v_.shape()) and k_.dim(.hd) == q_.dim(.hd) and q_.dim(.hg) > 0, "CuTe paged attention requires aligned local Q/KV heads and matching caches", .{});
    const decode = !parameters.options_.is_prefill;
    var c: kernel.Config = .{
        .head_dim = q_.dim(.hd),
        .tokens = q_.dim(.b),
        .batch = parameters.block_table.dim(.b),
        .kv_heads = k_.dim(.hkv),
        .group = q_.dim(.hg),
        .pages_per_seq = parameters.block_table.dim(.p),
        .page_size = k_.dim(.k_chunk),
        .max_query = if (decode) 1 else @min(q_.dim(.b), @as(i64, @intCast(parameters.options_.max_seqlen_q))),
        .dtype = zml.kernel.cute.from(q_.dtype()),
        .warps_per_query = if (decode) 4 else 2,
        .kv_tile = if (q_.dim(.hd) == 512) 32 else 64,
        .causal = opts.mask.isCausal(),
        .window = opts.mask.slidingWindow() orelse 0,
        .scale = opts.scale orelse 1 / @sqrt(@as(f32, @floatFromInt(q_.dim(.hd)))),
        .sink = opts.sink != null,
    };
    // Mixed batches also contain single-token rows. Partition their KV
    // range so one long decode cannot hold up all the prefill CTAs.
    const query_tiles = if (decode) c.batch * c.queryTiles() else c.batch + @divTrunc(c.tokens * c.group + c.rows() - 1, c.rows());
    const work = query_tiles * c.kv_heads;
    const bound = if (c.window > 0) @min(c.window, c.pages_per_seq * c.page_size) else c.pages_per_seq * c.page_size;
    c.splits = @max(1, @min(32, @min(@divTrunc(@as(i64, @intCast(cute.multiprocessorCount(zml.Compiler.current().platform))) * 2 + work - 1, work), @divTrunc(bound + 127, 128))));
    const sink = if (opts.sink) |s| s.convert(.f32).reshape(.{c.kv_heads * c.group}) else zml.Tensor.zeroes(.init(.{c.kv_heads * c.group}, .f32));
    const out = kernel.Kernel.call(.{
        .q = q_,
        .k = k_,
        .v = v_,
        .table = parameters.block_table,
        .lengths = parameters.seq_lens,
        .starts = parameters.query_start_len,
        .sink = sink,
    }, .{ .out = if (c.splits > 1) zml.Shape.init(.{ c.splits, c.tokens, c.kv_heads * c.group, c.head_dim + 2 }, .f32) else q_.shape() }, .{
        .cfg = c,
        .grid = .{ @intCast(c.batch * c.queryTiles()), @intCast(c.kv_heads), @intCast(c.splits) },
        .block = .{ 128, 1, 1 },
    }).out;
    const result = if (c.splits > 1) kernel.Merge.call(.{ .partials = out, .starts = parameters.query_start_len, .sink = sink }, .{ .out = q_.shape() }, .{
        .cfg = c,
        .grid = .{ @intCast(c.tokens), @intCast(c.kv_heads * c.group), 1 },
        .block = .{ 128, 1, 1 },
    }).out else out;
    return result.transpose(q.shape());
}
