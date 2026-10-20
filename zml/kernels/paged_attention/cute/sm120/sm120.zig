const stdx = @import("stdx");

const zml = @import("../../../../zml.zig");
const paged = @import("../../paged_attention.zig");
const triton = @import("../../triton/triton.zig");
const kernel = @import("attention.zig");

pub const Options = triton.paged.Options;
pub const Parameters = triton.paged.Parameters;

test {
    _ = kernel;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major == 12;
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
        var c: kernel.Config = .{
            .head_dim = q.dim(.hd),
            .tokens = q.dim(.b),
            .batch = p.block_table.dim(.b),
            .kv_heads = k.dim(.hkv),
            .group = q.dim(.hg),
            .pages_per_seq = p.block_table.dim(.p),
            .page_size = k.dim(.k_chunk),
            .max_query = if (decode) 1 else @min(q.dim(.b), @as(i64, @intCast(p.options_.max_seqlen_q))),
            .dtype = zml.kernel.cute.from(q.dtype()),
            .warps_per_query = if (decode) 4 else 2,
            .kv_tile = if (q.dim(.hd) == 512) 32 else 64,
            .causal = self.opts.mask.isCausal(),
            .window = self.opts.mask.slidingWindow() orelse 0,
            .scale = self.opts.scale orelse 1 / @sqrt(@as(f32, @floatFromInt(q.dim(.hd)))),
            .sink = self.opts.sink != null,
        };
        // Mixed batches also contain single-token rows. Partition their KV
        // range so one long decode cannot hold up all the prefill CTAs.
        const query_tiles = if (decode) c.batch * c.queryTiles() else c.batch + @divTrunc(c.tokens * c.group + c.rows() - 1, c.rows());
        const work = query_tiles * c.kv_heads;
        const bound = if (c.window > 0) @min(c.window, c.pages_per_seq * c.page_size) else c.pages_per_seq * c.page_size;
        c.splits = @max(1, @min(32, @min(@divTrunc(@as(i64, @intCast(triton.getCuCount())) * 2 + work - 1, work), @divTrunc(bound + 127, 128))));
        const sink = if (self.opts.sink) |s| s.convert(.f32).reshape(.{c.kv_heads * c.group}) else zml.Tensor.zeroes(.init(.{c.kv_heads * c.group}, .f32));
        const out = kernel.Kernel.call(.{
            .q = q,
            .k = k,
            .v = v,
            .table = p.block_table,
            .lengths = p.seq_lens,
            .starts = p.query_start_len,
            .sink = sink,
        }, .{ .out = if (c.splits > 1) zml.Shape.init(.{ c.splits, c.tokens, c.kv_heads * c.group, c.head_dim + 2 }, .f32) else q.shape() }, .{
            .cfg = c,
            .grid = .{ @intCast(c.batch * c.queryTiles()), @intCast(c.kv_heads), @intCast(c.splits) },
            .block = .{ 128, 1, 1 },
        }).out;
        const result = if (c.splits > 1) kernel.Merge.call(.{ .partials = out, .starts = p.query_start_len, .sink = sink }, .{ .out = q.shape() }, .{
            .cfg = c,
            .grid = .{ @intCast(c.tokens), @intCast(c.kv_heads * c.group), 1 },
            .block = .{ 128, 1, 1 },
        }).out else out;
        return result.transpose(output_shape);
    }
};
