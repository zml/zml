const zml = @import("zml/core");
const cute = @import("../cute.zig");
const common = @import("../common.zig");
const kernel = @import("attention.zig");
const Merge = @import("../Merge.zig");
const sm90 = @import("../sm90/sm90.zig");

pub const Options = cute.Options;
pub const Parameters = cute.Parameters;

test {
    _ = kernel;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    // Only sm80-level features (cp.async, ldmatrix, mma.sync), so Hopper runs it too.
    return cc.major == 9 or cc.major == 12;
}

// A 64-row tile halves K/V traffic per row but pads every sequence to 64 rows and
// has half as many CTAs, so it only pays when sequences are long and fill the GPU.
fn warpsPerQuery(decode: bool, head_dim: i64, tokens: i64, batch: i64, group: i64, kv_heads: i64, multiprocessors: i64) i64 {
    if (decode) {
        return 4;
    }

    const rows_per_sequence = @divTrunc(tokens * group, batch);
    const big_tile_ctas = @divTrunc(tokens * group, 64) * kv_heads;
    if (head_dim <= 128 and rows_per_sequence >= 256 and big_tile_ctas >= multiprocessors) {
        return 1;
    }

    return 2;
}

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: cute.AttentionOptions) zml.Tensor {
    if (sm90.isAvailable(zml.Compiler.current().platform) and sm90.supports(parameters, q, k, opts)) {
        return sm90.pagedAttention(parameters, q, k, v, opts);
    }

    const q_, const k_, const v_ = common.pagedOperands(q, k, v);
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
        .warps_per_query = warpsPerQuery(decode, q_.dim(.hd), q_.dim(.b), parameters.block_table.dim(.b), q_.dim(.hg), k_.dim(.hkv), common.multiprocessorCount()),
        .kv_tile = if (q_.dim(.hd) == 512) 32 else 64,
        .causal = opts.mask.isCausal(),
        .window = opts.mask.slidingWindow() orelse 0,
        .scale = opts.scale orelse 1 / @sqrt(@as(f32, @floatFromInt(q_.dim(.hd)))),
        .sink = opts.sink != null,
        .separate_v = q_.dim(.hd) <= 256,
    };
    // Mixed batches also contain single-token rows. Partition their KV
    // range so one long decode cannot hold up all the prefill CTAs.
    const query_tiles = if (decode) c.batch * c.queryTiles() else common.prefillQueryTiles(c, c.rows());
    c.splits = common.splitCount(c, common.multiprocessorCount(), query_tiles, common.keyBound(c), 128);
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
        .block = .{ 128, 1, 1 },
    }).out;
    const result = Merge.mergeSplits(c, out, parameters.query_start_len, sink, q_.shape());
    return result.transpose(q.shape());
}
