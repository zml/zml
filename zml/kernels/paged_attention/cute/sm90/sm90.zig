const stdx = @import("stdx");

const zml = @import("zml/core");
const cute = @import("../cute.zig");
const kernel = @import("attention.zig");
const merge_kernel = @import("../sm120/attention.zig");

pub const Parameters = cute.Parameters;

test {
    _ = kernel;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major == 9;
}

// Short key ranges do not amortize the cp.async prologue and the wgmma pipeline fill, where the mma.sync kernel is faster.
const minimum_key_range = 1024;

pub fn supports(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, opts: cute.AttentionOptions) bool {
    const head_dim = q.dim(.hd);
    const key_range = parameters.block_table.dim(.p) * k.dim(.k_chunk);
    return parameters.options_.is_prefill and opts.mask != .indices and (head_dim == 64 or head_dim == 128) and key_range >= minimum_key_range;
}

// A grid that already fits in one wave gains nothing from splitting: it only adds a second wave, fp32 partials and the merge kernel.
fn splitCount(multiprocessors: i64, work: i64, grid_ctas: i64, bound: i64) i64 {
    if (grid_ctas <= multiprocessors) {
        return 1;
    }

    return @max(1, @min(32, @min(@divTrunc(multiprocessors * 2 + work - 1, work), @divTrunc(bound + kernel.block_keys - 1, kernel.block_keys))));
}

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: cute.AttentionOptions) zml.Tensor {
    const q_ = q.transpose(.{ .b, .hkv, .hg, .hd });
    const k_ = k.transpose(.{ .page, .k_chunk, .hkv, .hd });
    const v_ = v.transpose(.{ .page, .k_chunk, .hkv, .hd });
    stdx.debug.assert(q_.dim(.hkv) == k_.dim(.hkv) and k_.shape().eql(v_.shape()) and k_.dim(.hd) == q_.dim(.hd) and q_.dim(.hg) > 0, "CuTe paged attention requires aligned local Q/KV heads and matching caches", .{});
    var c: kernel.Config = .{
        .head_dim = q_.dim(.hd),
        .tokens = q_.dim(.b),
        .batch = parameters.block_table.dim(.b),
        .kv_heads = k_.dim(.hkv),
        .group = q_.dim(.hg),
        .pages_per_seq = parameters.block_table.dim(.p),
        .page_size = k_.dim(.k_chunk),
        .max_query = @min(q_.dim(.b), @as(i64, @intCast(parameters.options_.max_seqlen_q))),
        .dtype = zml.kernel.cute.from(q_.dtype()),
        .causal = opts.mask.isCausal(),
        .window = opts.mask.slidingWindow() orelse 0,
        .scale = opts.scale orelse 1 / @sqrt(@as(f32, @floatFromInt(q_.dim(.hd)))),
        .sink = opts.sink != null,
        .splits = 1,
    };
    const query_tiles = c.batch + @divTrunc(c.tokens * c.group + kernel.block_rows - 1, kernel.block_rows);
    const work = query_tiles * c.kv_heads;
    const bound = if (c.window > 0) @min(c.window, c.pages_per_seq * c.page_size) else c.pages_per_seq * c.page_size;
    const multiprocessors: i64 = @intCast(cute.multiprocessorCount(zml.Compiler.current().platform));
    c.splits = splitCount(multiprocessors, work, c.batch * c.queryTiles() * c.kv_heads, bound);
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
        .block = .{ kernel.thread_count, 1, 1 },
    }).out;
    const merge_config: merge_kernel.Config = .{
        .head_dim = c.head_dim,
        .tokens = c.tokens,
        .batch = c.batch,
        .kv_heads = c.kv_heads,
        .group = c.group,
        .pages_per_seq = c.pages_per_seq,
        .page_size = c.page_size,
        .max_query = c.max_query,
        .dtype = c.dtype,
        .warps_per_query = 2,
        .kv_tile = 64,
        .causal = c.causal,
        .window = c.window,
        .scale = c.scale,
        .sink = c.sink,
        .separate_v = false,
        .splits = c.splits,
    };
    const result = if (c.splits > 1) merge_kernel.Merge.call(.{ .partials = out, .starts = parameters.query_start_len, .sink = sink }, .{ .out = q_.shape() }, .{
        .cfg = merge_config,
        .grid = .{ @intCast(c.tokens), @intCast(c.kv_heads * c.group), 1 },
        .block = .{ 128, 1, 1 },
    }).out else out;
    return result.transpose(q.shape());
}
