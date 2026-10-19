const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const paged = @import("../paged_attention.zig");
const triton = @import("../triton/triton.zig");
const kernel = @import("attention.zig");

pub const Options = triton.paged.Options;
pub const Parameters = triton.paged.Parameters;

test {
    _ = kernel;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major == 10;
}

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: paged.AttentionOptions) zml.Tensor {
    if (!isAvailable(zml.Compiler.current().platform) or (q.dim(.hd) != 512 and q.dim(.hd) != 256) or
        (q.dtype() != .bf16 and q.dtype() != .f16) or opts.mask == .indices)
        return triton.paged.pagedAttention(parameters, q, k, v, opts);
    stdx.debug.assert(q.dtype() == k.dtype() and q.dtype() == v.dtype(), "CuTe paged attention requires matching Q/K/V types", .{});
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
        const all_decode = !p.options_.is_prefill;
        const sms: i64 = @intCast(triton.getCuCount());
        const rows = kernel.decodeRows(q.dim(.hg));
        const head_tiles = @divTrunc(q.dim(.hg) + rows - 1, rows);
        const max_splits = if (all_decode) @min(128, @divTrunc(sms * 2 + k.dim(.hkv) * head_tiles - 1, k.dim(.hkv) * head_tiles)) else 1;
        const window = self.opts.mask.slidingWindow() orelse 0;
        const large_decode_tile = all_decode and q.dim(.hd) == 256 and
            q.dim(.hg) <= 4 and window >= 128 and window <= 1024 and
            p.block_table.dim(.b) * k.dim(.hkv) * head_tiles >= @divTrunc(sms, 2);
        const kv_tile: i64 = if (large_decode_tile) 128 else 64;
        const decode_work = p.block_table.dim(.b) * k.dim(.hkv) * head_tiles;
        const decode_waves: i64 = if (large_decode_tile and decode_work < sms * 2) 4 else 2;
        // A full decode batch supplies enough independent sequence/head CTAs
        // to stream the window without split scratch or a separate merge.
        // Smaller grids retain the existing partitioned path.
        const max_window_tiles = @divTrunc(@as(i64, window) + 2 * kv_tile - 2, kv_tile);
        const tiles_per_split: i64 = if (!large_decode_tile and all_decode and q.dim(.hd) == 256 and window > 0) @max(1, @divTrunc(max_window_tiles + 127, 128)) else 0;
        const splits = if (tiles_per_split > 0)
            @divTrunc(max_window_tiles + tiles_per_split - 1, tiles_per_split)
        else if (all_decode)
            @min(max_splits, @max(1, @divTrunc(sms * decode_waves + p.block_table.dim(.b) * k.dim(.hkv) * head_tiles - 1, p.block_table.dim(.b) * k.dim(.hkv) * head_tiles)))
        else
            max_splits;
        // Keep independent split CTAs for low-occupancy shapes. At full
        // occupancy, a fixed number of workers reuses Q and TMEM and avoids
        // launching the entire window bound for short runtime contexts.
        const split_stride: i64 = if (tiles_per_split == 1 and splits > 2 and
            p.block_table.dim(.b) * k.dim(.hkv) * head_tiles >= sms * 2) 2 else 0;
        const cfg: kernel.Config = .{
            .head_dim = q.dim(.hd),
            .kv_tile = kv_tile,
            // Prefetch an extra feature stage for non-split D512 decode;
            // split workloads retain the smaller shared-memory footprint.
            .load_buffers = if (kv_tile == 128 or (all_decode and q.dim(.hd) == 512 and splits == 1)) 3 else 2,
            .tokens = q.dim(.b),
            .batch = p.block_table.dim(.b),
            .kv_heads = k.dim(.hkv),
            .group = q.dim(.hg),
            .pages_per_seq = p.block_table.dim(.p),
            .page_size = k.dim(.k_chunk),
            .max_query = if (all_decode) 1 else @min(q.dim(.b), @as(i64, @intCast(p.options_.max_seqlen_q))),
            .dtype = zml.kernel.cute.from(q.dtype()),
            .all_decode = all_decode,
            .causal = self.opts.mask.isCausal(),
            .window = window,
            .scale = self.opts.scale orelse (1 / @sqrt(@as(f32, @floatFromInt(q.dim(.hd))))),
            .sink = self.opts.sink != null,
            .sms = sms,
            .splits = @max(1, splits),
            .tiles_per_split = tiles_per_split,
            .split_stride = split_stride,
        };
        const sink = if (self.opts.sink) |s| s.convert(.f32).reshape(.{cfg.kv_heads * cfg.group}) else zml.Tensor.zeroes(.init(.{cfg.kv_heads * cfg.group}, .f32));
        const out_shape = if (cfg.splits > 1) zml.Shape.init(.{ cfg.splits, cfg.tokens, cfg.kv_heads * cfg.group, cfg.head_dim + 2 }, .f32) else q.shape();
        const out = kernel.Kernel.call(.{
            .q = q,
            .k = k,
            .v = v,
            .table = p.block_table,
            .lengths = p.seq_lens,
            .starts = p.query_start_len,
            .sink = sink,
        }, .{ .out = out_shape }, .{
            .cfg = cfg,
            .grid = .{ @intCast(cfg.batch * cfg.queryTiles() * (if (cfg.headsInX()) cfg.kv_heads else 1)), @intCast(if (cfg.headsInX()) 1 else cfg.kv_heads * cfg.outputTiles()), @intCast(if (cfg.split_stride > 0) cfg.split_stride else cfg.splits) },
            .block = kernel.block,
        }).out;
        const result = if (cfg.splits > 1) kernel.Merge.call(.{
            .partials = out,
            .lengths = p.seq_lens,
            .starts = p.query_start_len,
            .sink = sink,
        }, .{ .out = q.shape() }, .{ .cfg = cfg, .grid = .{ @intCast(cfg.kv_heads * cfg.group), @intCast(cfg.batch), 1 }, .block = .{ @intCast(cfg.head_dim), 1, 1 } }).out else out;
        return result.transpose(output_shape);
    }
};
