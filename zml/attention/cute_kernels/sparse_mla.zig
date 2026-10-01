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
//! it takes the paged caches with their physical rows already resolved from the page
//! tables, and runs the kernel once per group of 16 heads.
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
    scale: f64 = 0.04419417382415922, // 1 / sqrt(512)
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

/// Shapes (with dtypes) of the caller's attention inputs, before any flattening.
/// The last axis is the per-row axis: the latent width for Q and the value
/// caches, the scale bytes per row for the scale caches.
pub const Layout = struct {
    q: zml.Shape, // [.., heads, 512] bf16
    window_values: zml.Shape, // [.., 512] f8e4m3fn
    window_scales: zml.Shape, // [.., 16] f8e8m0
    compressed_values: ?zml.Shape, // [.., 512] f4e2m1, null for window-only layers
    compressed_scales: ?zml.Shape, // [.., 32] f8e4m3fn
    window_capacity: i64, // window rows per query
    compressed_capacity: i64, // compressed (top-k) rows per query, 0 for window-only layers
};

/// Whether the kernel supports these inputs on this platform: Blackwell, BF16 Q with
/// a 512-wide latent and a multiple of 16 heads (callers split them into groups of
/// 16, one `forward` each), and the DeepSeek-V4.1 cache formats (FP8 window rows
/// with E8M0 scales per 32 values, packed FP4 compressed rows with E4M3 scales per
/// 16), a window of 128 rows and at most 512 compressed rows per query.
pub fn supports(platform: *const zml.Platform, l: Layout) bool {
    const last = struct {
        fn f(shape: zml.Shape) i64 {
            return shape.dim(shape.rank() - 1);
        }
    }.f;
    const heads = if (l.q.rank() >= 2) l.q.dim(l.q.rank() - 2) else 0;
    return isAvailable(platform) and
        l.q.dtype() == .bf16 and last(l.q) == 512 and heads > 0 and @rem(heads, 16) == 0 and heads <= 64 and
        l.window_values.dtype() == .f8e4m3fn and last(l.window_values) == 512 and
        l.window_scales.dtype() == .f8e8m0 and last(l.window_scales) == 16 and
        (l.compressed_capacity == 0 or compressedFormat(l.compressed_values, l.compressed_scales)) and
        l.window_capacity == 128 and
        @rem(l.compressed_capacity, 128) == 0 and l.compressed_capacity >= 0 and l.compressed_capacity <= 512;
}

fn compressedFormat(values: ?zml.Shape, scales: ?zml.Shape) bool {
    const v = values orelse return false;
    const sc = scales orelse return false;
    return v.dtype() == .f4e2m1 and v.dim(v.rank() - 1) == 512 and
        sc.dtype() == .f8e4m3fn and sc.dim(sc.rank() - 1) == 32;
}

pub fn forward(c: Config, a: Inputs) zml.Tensor {
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

/// Whether the kernel supports sparse MLA of `q` [.q, .h, .hd] over these caches: quantized
/// caches (no global scale) in the formats and sizes of `supports`, on Blackwell.
pub fn supportsInputs(q: Tensor, cache: Mla.Cache, compressed: ?Mla.Cache) bool {
    const window = quantized(cache) orelse return false;
    const compressed_input: ?zml.quantization.QuantizedInput = if (compressed) |c| quantized(c) orelse return false else null;
    return supports(zml.Compiler.current().platform, .{
        .q = q.shape(),
        .window_values = window.values.shape(),
        .window_scales = window.scales.shape(),
        .compressed_values = if (compressed_input) |c| c.values.shape() else null,
        .compressed_scales = if (compressed_input) |c| c.scales.shape() else null,
        .window_capacity = cache.positions.dim(.topk),
        .compressed_capacity = if (compressed) |c| c.positions.dim(.topk) else 0,
    });
}

/// Backend `.cute` of `paged_attention.Mla.pagedSparseAttention`: Q [.q, .h, .hd], a
/// quantized window cache and optionally a quantized compressed cache, `rows` /
/// `compressed_rows` the physical rows of their selected positions. Panics on inputs the
/// kernel does not support (see `supportsInputs`). Each stream is scanned up to its last valid entry.
pub fn pagedAttention(q: Tensor, cache: Mla.Cache, rows: Tensor, compressed: ?Mla.Cache, compressed_rows: ?Tensor, sink: ?Tensor, active_count: Tensor, opts: Mla.Options) Tensor {
    if (!supportsInputs(q, cache, compressed)) std.debug.panic("CuTe sparse MLA does not support q {f} over a {s} cache of {} rows per query and {s}: use another backend", .{
        q.shape(),
        @tagName(cache.storage),
        cache.positions.dim(.topk),
        if (compressed) |c| @tagName(c.storage) else "no compressed cache",
    });
    const attention_sink = sink orelse std.debug.panic("CuTe sparse MLA requires an attention sink", .{});
    if (opts.value_rank != q.dim(.hd)) std.debug.panic("CuTe sparse MLA requires value_rank ({}) == head dim ({})", .{ opts.value_rank, q.dim(.hd) });
    const window = cache.storage.quantized;
    const compressed_input: ?zml.quantization.QuantizedInput = if (compressed) |c| c.storage.quantized else null;
    const head_dim: f64 = @floatFromInt(q.dim(.hd));
    return zml.ops.manualComputation(Shard.run, Shard{
        .q = q.rename(.{ .q = .b }),
        .window_values = window.values,
        .window_scales = window.scales,
        .window_indices = rows,
        .compressed_values = if (compressed_input) |c| c.values else null,
        .compressed_scales = if (compressed_input) |c| c.scales else null,
        // Window-only: a placeholder, not read (compressed capacity 0).
        .compressed_indices = compressed_rows orelse rows,
        .compressed_capacity = if (compressed_rows) |r| r.dim(.topk) else 0,
        .sink = attention_sink,
        .active_count = active_count,
        .scale = if (opts.scale) |scale| scale else 1 / @sqrt(head_dim),
    }, q.shape().rename(.{ .q = .b })).rename(.{ .b = .q });
}

/// The quantized storage of `cache` (one row per token, no global scale), or null.
fn quantized(cache: Mla.Cache) ?zml.quantization.QuantizedInput {
    if (cache.storage != .quantized) return null;
    const input = cache.storage.quantized;
    if (input.global_scale != null) return null;
    return input;
}

/// Per-device part: this shard's heads, run in groups of 16 (a TP4 shard has exactly 16).
const Shard = struct {
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

    fn run(self: Shard, _: zml.Shape) Tensor {
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
            outs[g] = forward(cfg, .{
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
