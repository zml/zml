const zml = @import("zml/core");
const cute = @import("../cute.zig");
const common = @import("../common.zig");
const kernel = @import("attention.zig");
const Merge = @import("../Merge.zig");

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
fn splitCount(c: kernel.Config, multiprocessors: i64) i64 {
    if (c.batch * c.queryTiles() * c.kv_heads <= multiprocessors) {
        return 1;
    }

    return common.splitCount(c, multiprocessors, common.prefillQueryTiles(c, kernel.block_rows), common.keyBound(c), kernel.block_keys);
}

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: cute.AttentionOptions) zml.Tensor {
    const q_, const k_, const v_ = common.pagedOperands(q, k, v);
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
    c.splits = splitCount(c, common.multiprocessorCount());
    const sink = common.sinkOperand(opts, c.kv_heads * c.group);
    const out = kernel.Kernel.call(.{
        .q = q_,
        .k = k_,
        .v = v_,
        .table = parameters.block_table,
        .lengths = parameters.seq_lens,
        .starts = parameters.query_start_len,
        .sink = sink,
    }, .{ .out = common.splitOutputShape(c, q_.shape()) }, .{
        .cfg = c,
        .grid = .{ @intCast(c.batch * c.queryTiles()), @intCast(c.kv_heads), @intCast(c.splits) },
        .block = .{ kernel.thread_count, 1, 1 },
    }).out;
    const result = Merge.mergeSplits(c, out, parameters.query_start_len, sink, q_.shape());
    return result.transpose(q.shape());
}
