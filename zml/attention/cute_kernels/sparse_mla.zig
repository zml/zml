//! CuTe backend of sparse MLA (`paged_attention.Mla`, backend `.cute`) on Blackwell
//! (SM100/SM103): in-place sparse attention over DeepSeek-V4.1-style quantized caches.
//!
//! Q is BF16 [queries, heads, 512]. Keys and values are the same 512-wide latent
//! rows, selected per query from two quantized paged caches: a sliding window
//! cache (FP8 e4m3, E8M0 scale per 32 values) and a compressed cache (packed
//! FP4 e2m1, E4M3 scale per 16 values). The selected rows are read and
//! dequantized on chip (`flashmla_sm100.zig`): nothing is gathered in global
//! memory and there is no split-KV reduction.
//!
//! `pagedAttention` is the backend entry of `paged_attention.Mla.pagedSparseAttention`:
//! it resolves the selected positions to physical rows of the paged caches and runs the
//! kernel once per group of 16 heads.
const std = @import("std");

const zml = @import("../../zml.zig");
const Mla = @import("../paged_attention.zig").Mla;
const Tensor = zml.Tensor;

pub const Config = struct {
    queries: i64,
    window_capacity: i64,
    compressed_capacity: i64,
    window_slots: i64,
    compressed_slots: i64,
    heads: i64 = 16,
    scale: f64 = 1.0 / std.math.sqrt(512.0),
    /// CTAs per query token (power of two); null picks one from the SM count.
    cluster: ?i64 = null,

    /// Enough CTAs to cover the SMs in a single wave, but no more than there
    /// are key blocks (64 candidates each) and at most 8 (portable cluster size).
    /// Clusters only co-schedule within a GPC, so large clusters do not fill
    /// every SM: on GB300 (152 SMs, one 226 KiB CTA per SM) at most 14 clusters
    /// of 8 (112 CTAs) and 36 clusters of 4 run at once (measured 2026-09-29).
    /// Persistent CTAs (one per SM: one 226 KiB CTA fits) when tokens are not split;
    /// larger batches and prefill loop over tokens inside the kernel.
    pub fn persistentCtas(c: Config, sm_count: usize) i64 {
        if (c.clusterSize(sm_count) > 1) return c.queries;
        return @min(c.queries, @as(i64, @intCast(sm_count)));
    }

    pub fn clusterSize(c: Config, sm_count: usize) i64 {
        if (c.cluster) |n| return n;
        // Window-only layers have at most two blocks: splitting them never pays for the merge.
        if (c.compressed_capacity == 0) return 1;
        const blocks = @divTrunc(c.window_capacity + 63, 64) + @divTrunc(c.compressed_capacity + 63, 64);
        const sms: i64 = @intCast(sm_count);
        var size: i64 = 8;
        while (size > 1) : (size = @divExact(size, 2)) {
            const resident = switch (size) {
                8 => @divTrunc(sms * 112, 152),
                4 => @divTrunc(sms * 144, 152),
                else => sms,
            };
            if (size <= blocks and c.queries * size <= resident) break;
        }
        return size;
    }
};

/// Kernel inputs: flat cache rows and per-query physical row indices.
pub const Inputs = struct {
    q: zml.Tensor, // bf16 [queries, heads, 512]
    window_values: zml.Tensor, // u8 [window_slots, 512] e4m3
    window_scales: zml.Tensor, // u8 [window_slots, 16] e8m0
    compressed_values: zml.Tensor, // u8 [compressed_slots, 256] e2m1 x2 (even value in the low nibble)
    compressed_scales: zml.Tensor, // u8 [compressed_slots, 32] e4m3
    window_indices: zml.Tensor, // i32 [queries, window_capacity], physical rows, <0 invalid
    compressed_indices: zml.Tensor, // i32 [queries, compressed_capacity]
    lengths: zml.Tensor, // i32 [queries, 2] scan bounds per stream
    sink: zml.Tensor, // f32 [heads] per-head logit with zero value
    active: zml.Tensor, // i32 [1] leading active query rows
};

const kernel = @import("flashmla_sm100.zig");

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major == 10;
}

/// Whether the kernel supports these inputs: Blackwell, BF16 Q with a 512-wide latent
/// and a multiple of 16 heads (at most 64; they run in groups of 16, one `kernelCall`
/// each), and the DeepSeek-V4.1 cache formats without a global scale: FP8 window rows
/// with E8M0 scales per 32 values, a window of 128 rows, and packed FP4 compressed rows
/// with E4M3 scales per 16 values, at most 512 (a multiple of 128) per query.
pub fn supports(args: Mla.SparseAttentionArgs) bool {
    if (!isAvailable(zml.Compiler.current().platform)) return false;
    const q = args.q.shape();
    const heads = q.dim(.h);
    if (q.dtype() != .bf16 or q.dim(.hd) != 512 or @rem(heads, 16) != 0 or heads > 64) return false;

    const window = switch (args.kv.cache) {
        .quantized => |input| input,
        .latent => return false,
    };
    if (window.global_scale != null) return false;
    if (window.values.dtype() != .f8e4m3fn or lastDim(window.values) != 512) return false;
    if (window.scales.dtype() != .f8e8m0 or lastDim(window.scales) != 16) return false;
    if (args.kv.positions.dim(.topk) != 128) return false;

    const compressed = args.compressed orelse return true;
    const input = switch (compressed.cache) {
        .quantized => |input| input,
        .latent => return false,
    };
    if (input.global_scale != null) return false;
    if (input.values.dtype() != .f4e2m1 or lastDim(input.values) != 512) return false;
    if (input.scales.dtype() != .f8e4m3fn or lastDim(input.scales) != 32) return false;
    const capacity = compressed.positions.dim(.topk);
    return @rem(capacity, 128) == 0 and capacity <= 512;
}

fn lastDim(t: Tensor) i64 {
    return t.dim(t.rank() - 1);
}

/// One kernel launch over 16 heads (see `Config` and `Inputs`).
pub fn kernelCall(c: Config, a: Inputs) zml.Tensor {
    // The kernel is compiled for these shapes (see `supports`); anything else would
    // read the caches with the wrong layout.
    std.debug.assert(c.heads == 16);
    std.debug.assert(c.window_capacity == 128);
    std.debug.assert(@rem(c.compressed_capacity, 128) == 0 and c.compressed_capacity <= 512);
    std.debug.assert(a.q.dtype() == .bf16 and a.q.dim(1) == 16 and a.q.dim(2) == 512);
    std.debug.assert(a.window_values.dim(1) == 512 and a.window_scales.dim(1) == 16);
    std.debug.assert(a.compressed_values.dim(1) == 256 and a.compressed_scales.dim(1) == 32);
    const out = zml.Shape.init(.{ c.queries, c.heads, 512 }, .bf16);
    return kernel.Program.call(.{
        .q = a.q,
        .wv = a.window_values,
        .ws = a.window_scales,
        .cv = a.compressed_values,
        .cs = a.compressed_scales,
        .wi = a.window_indices,
        .ci = a.compressed_indices,
        .lengths = a.lengths,
        .sink = a.sink,
        .active = a.active,
    }, .{ .out = out }, .{ .cfg = .{
        .queries = c.queries,
        .window_capacity = c.window_capacity,
        .compressed_capacity = c.compressed_capacity,
        .window_slots = c.window_slots,
        .compressed_slots = c.compressed_slots,
        .scale = c.scale,
        .cluster = c.clusterSize(zml.attention.triton.getCuCount()),
        .ctas = c.persistentCtas(zml.attention.triton.getCuCount()),
    } }).out;
}

/// Backend `.cute` of `paged_attention.Mla.pagedSparseAttention`: the selected rows of a
/// quantized window cache and optionally of a quantized compressed cache, read in place.
/// Panics on inputs the kernel does not support (see `supports`). Each stream is scanned
/// up to its last valid entry.
pub fn pagedAttention(args: Mla.SparseAttentionArgs, opts: Mla.Options) Tensor {
    if (!supports(args)) std.debug.panic("CuTe sparse MLA does not support q {f} over a {s} cache of {} rows per query and {s}: use another backend", .{
        args.q.shape(),
        @tagName(args.kv.cache),
        args.kv.positions.dim(.topk),
        if (args.compressed) |c| @tagName(c.cache) else "no compressed cache",
    });
    const sink = args.sink orelse std.debug.panic("CuTe sparse MLA requires an attention sink", .{});
    if (opts.value_rank != args.q.dim(.hd)) std.debug.panic("CuTe sparse MLA requires value_rank ({}) == head dim ({})", .{ opts.value_rank, args.q.dim(.hd) });
    // Physical rows of the selected positions, from each selection's page table.
    const active_count = args.kv.activeCount();
    const rows = args.kv.physicalRows(args.tokens_pos, active_count);
    const window = args.kv.cache.quantized;
    const head_dim: f64 = @floatFromInt(args.q.dim(.hd));
    var context: ShardContext = .{
        .q = args.q.rename(.{ .q = .b }),
        .window_values = window.values,
        .window_scales = window.scales,
        .window_indices = rows,
        .compressed_values = null,
        .compressed_scales = null,
        // Window-only: a placeholder, not read (compressed capacity 0).
        .compressed_indices = rows,
        .compressed_capacity = 0,
        .sink = sink,
        .active_count = active_count,
        .scale = if (opts.scale) |scale| scale else 1 / @sqrt(head_dim),
    };
    if (args.compressed) |c| {
        const compressed_rows = c.physicalRows(args.tokens_pos, active_count);
        context.compressed_values = c.cache.quantized.values;
        context.compressed_scales = c.cache.quantized.scales;
        context.compressed_indices = compressed_rows;
        context.compressed_capacity = compressed_rows.dim(.topk);
    }
    return zml.ops.manualComputation(ShardContext.run, context, args.q.shape().rename(.{ .q = .b })).rename(.{ .b = .q });
}

/// Per-device part: this shard's heads, run in groups of 16 (a TP4 shard has exactly 16).
const ShardContext = struct {
    q: Tensor,
    window_values: Tensor,
    window_scales: Tensor,
    window_indices: Tensor,
    compressed_values: ?Tensor, // null for window-only layers
    compressed_scales: ?Tensor,
    compressed_indices: Tensor,
    compressed_capacity: i64,
    sink: Tensor,
    active_count: Tensor,
    scale: f64,

    fn run(self: ShardContext, _: zml.Shape) Tensor {
        const q = self.q;
        const heads = q.dim(.h);
        const queries = q.dim(.b);
        std.debug.assert(@rem(heads, 16) == 0);
        const window_rows: i64 = @intCast(self.window_values.shape().count() / 512);
        const compressed_rows: i64 = if (self.compressed_values) |v| @intCast(v.shape().count() / 512) else 1;
        // Window-only layers: one zero row as a placeholder (compressed capacity 0, never read).
        const compressed_values = if (self.compressed_values) |v| v.reshape(.{ compressed_rows, 256, 2 }).bitCast(.u8) else Tensor.zeroes(zml.Shape.init(.{ 1, 256 }, .u8));
        const compressed_scales = if (self.compressed_scales) |sc| sc.bitCast(.u8).reshape(.{ compressed_rows, 32 }) else Tensor.zeroes(zml.Shape.init(.{ 1, 32 }, .u8));
        const window_capacity = self.window_indices.dim(.topk);
        const window_indices = self.window_indices.convert(.i32);
        const compressed_indices = if (self.compressed_capacity > 0)
            self.compressed_indices.convert(.i32)
        else
            Tensor.scalar(-1, .i32).broad(zml.Shape.init(.{ .q = queries, .topk = 0 }, .i32));
        const lengths = Tensor.concatenate(&.{
            scanLength(window_indices).insertAxes(.q, .{.stream}),
            scanLength(compressed_indices).insertAxes(.q, .{.stream}),
        }, .stream).transpose(.{ .q, .stream }).reshape(.{ queries, 2 });
        const cfg: Config = .{
            .queries = queries,
            .window_capacity = window_capacity,
            .compressed_capacity = self.compressed_capacity,
            .window_slots = window_rows,
            .compressed_slots = compressed_rows,
            .scale = self.scale,
        };
        const q_all = q.reshape(.{ queries, heads, 512 });
        const sink_all = self.sink.convert(.f32).reshape(.{heads});
        var outs: [4]Tensor = undefined;
        const groups: usize = @intCast(@divExact(heads, 16));
        for (0..groups) |g| {
            const start: i64 = @intCast(16 * g);
            outs[g] = kernelCall(cfg, .{
                .q = if (groups == 1) q_all else q_all.slice(1, .{ .start = start, .end = start + 16 }),
                .window_values = self.window_values.bitCast(.u8).reshape(.{ window_rows, 512 }),
                .window_scales = self.window_scales.bitCast(.u8).reshape(.{ window_rows, 16 }),
                .compressed_values = compressed_values,
                .compressed_scales = compressed_scales,
                .window_indices = window_indices.reshape(.{ queries, window_capacity }),
                .compressed_indices = compressed_indices.reshape(.{ queries, self.compressed_capacity }),
                .lengths = lengths,
                .sink = if (groups == 1) sink_all else sink_all.slice(0, .{ .start = start, .end = start + 16 }),
                .active = self.active_count.convert(.i32).reshape(.{1}),
            });
        }
        const out = if (groups == 1) outs[0] else Tensor.concatenate(outs[0..groups], 1);
        return out.withTags(.{ .b, .h, .hd }).withTags(q.shape());
    }
};

/// Rows to scan per query: through the last valid (>= 0) entry ([.q] i32).
fn scanLength(indices: Tensor) Tensor {
    return Tensor.iota(indices.shape(), .topk).addConstant(1)
        .mask(indices.cmp(.GE, .scalar(0, .i32)), 0).max(.topk).squeeze(.topk);
}
