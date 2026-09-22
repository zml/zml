//! CuTe boundary kernels around the persistent MXFP4 GEMMs of
//! `persistent_mxfp4.zig`: routed MXFP8 input quantization and the top-k
//! reduction.
const std = @import("std");

const zml = @import("../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;

pub const GroupedQuantConfig = struct {
    tokens: i64,
    hidden: i64,
    topk: i64,
    /// Rows stored per expert group.
    group_size: i64,
    capacity: i64,
    /// Route `r` goes to row `r * group_size`; `route_map` is not read.
    direct: bool = false,
};

pub const CombineConfig = struct {
    tokens: i64,
    hidden: i64,
    topk: i64,
    columns_per_cta: i64,
};

const GroupedQuantize = cute.Kernel(GroupedQuantConfig, .{
    .name = "mxfp4_cute_grouped_quantize",
    .inputs = &.{ "x", "route_map" },
    .outputs = &.{ "q", "s" },
    .run = groupedQuantizeKernel,
});

const Combine = cute.Kernel(CombineConfig, .{
    .name = "mxfp4_cute_combine",
    .inputs = &.{"d"},
    .outputs = &.{"y"},
    .run = combineKernel,
});

/// Quantize each token once, then scatter it to the physical rows assigned by
/// the routing. Scale bytes use the 128x4 layout consumed by the SM100
/// block-scaled MMA scale-factor tensor map.
pub fn groupedQuantize(x: zml.Tensor, route_map: zml.Tensor, cfg: GroupedQuantConfig) struct { q: zml.Tensor, s: zml.Tensor } {
    std.debug.assert(cfg.group_size == 16 or cfg.group_size == 32 or (cfg.direct and cfg.group_size == 1));
    const result = GroupedQuantize.call(
        .{ .x = x, .route_map = route_map },
        .{
            .q = .init(.{ cfg.capacity * cfg.group_size, cfg.hidden }, .f8e4m3fn),
            .s = .init(.{ cfg.capacity, @divExact(cfg.hidden, 32) * 128 }, .u8),
        },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(cfg.hidden, 4 * 128)), @intCast(cfg.tokens), 1 },
            .block = .{ 128, 1, 1 },
        },
    );
    return .{ .q = result.q, .s = result.s };
}

/// Amortize launch overhead for large prefills while retaining enough CTAs
/// to saturate the GPU for decode and small batches.
pub fn combineColumns(tokens: i64, hidden: i64) i64 {
    const elements = tokens * hidden;
    if (tokens > 1 and elements >= 512 * 2048) return 2048;
    if (tokens > 1 and elements >= 512 * 1024) return 1024;
    // Decode: one column per thread, like Python's finalize.
    if (tokens <= 8) return 128;
    return 256;
}

/// Sum the `[tokens * topk, hidden]` FP32 route rows of each token into BF16.
/// Routing weights were already applied before the down projection.
pub fn combine(d: zml.Tensor, cfg: CombineConfig) zml.Tensor {
    return Combine.call(
        .{ .d = d },
        .{ .y = .init(.{ cfg.tokens, cfg.hidden }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divTrunc(cfg.hidden + cfg.columns_per_cta - 1, cfg.columns_per_cta)), @intCast(cfg.tokens), 1 },
            .block = .{ 128, 1, 1 },
        },
    ).y;
}

/// Python `prepare_input_vector`: each thread quantizes four consecutive
/// values and stores them as one packed word, so eight lanes cover one
/// 32-value MX block.
fn groupedQuantizeKernel(b: *B, cfg: GroupedQuantConfig) cute.FinishError!void {
    std.debug.assert(@rem(cfg.hidden, 4 * 128) == 0);
    const a = try b.declareArgs(.{
        .x = .{ .tensor = .{ .dtype = .bf16, .shape = &.{ cfg.tokens, cfg.hidden } } },
        .route_map = .{ .tensor = .{ .dtype = .i32, .shape = &.{cfg.tokens * cfg.topk} } },
        // Four FP8 values per word.
        .q = .{ .tensor = .{ .dtype = .i32, .shape = &.{ cfg.capacity * cfg.group_size, @divExact(cfg.hidden, 4) } } },
        .s = .{ .tensor = .{ .dtype = .i8, .shape = &.{ cfg.capacity, @divExact(cfg.hidden, 32) * 128 } } },
    });

    // The persistent GEMM is allowed to launch as soon as this producer has
    // consumed its inputs. Its PDL dependency still prevents reading outputs
    // before this grid finishes writing them.
    b.launchDependents();
    const tid = b.threadIdx().x;
    const token = b.blockIdx().y;
    const word = b.blockIdx().x.mul(128).add(tid);
    var x: [4]cute.Value = undefined;
    for (&x, 0..) |*value, i| value.* = a.x.get(.{ token, word.mul(4).add(@as(i32, @intCast(i))) }).to(.f32);

    var maximum = x[0].abs().maximum(x[1].abs()).maximum(x[2].abs().maximum(x[3].abs()));
    inline for (.{ 1, 2, 4 }) |offset| {
        maximum = maximum.maximum(b.shuffleXor(maximum, offset));
    }
    const raw_bits = maximum.maximum(1e-4).mul(1.0 / 448.0).bitCast(.i32);
    const exponent = raw_bits.shrLogical(23).add(raw_bits.bitAnd(0x7fffff).ne(0).to(.i32));
    const inverse_scale = b.cst(.i32, 254).sub(exponent).shl(23).bitCast(.f32);
    const packed_word = b.packFp8x4(x[0].mul(inverse_scale), x[1].mul(inverse_scale), x[2].mul(inverse_scale), x[3].mul(inverse_scale));

    const scale_col = word.div(8);
    for (0..@intCast(cfg.topk)) |slot| {
        const route = token.mul(cfg.topk).add(slot);
        const row = if (cfg.direct) route.mul(cfg.group_size) else a.route_map.get(.{route});
        a.q.set(.{ row, word }, packed_word);

        var leader = b.openIf(tid.rem(8).eq(0));
        const scale_row = row.div(cfg.group_size);
        const row_in_group = row.rem(cfg.group_size);
        const swizzled_col = scale_col.div(4).mul(512)
            .add(row_in_group.rem(32).mul(16))
            .add(row_in_group.div(32).mul(4))
            .add(scale_col.rem(4));
        a.s.set(.{ scale_row, swizzled_col }, exponent.to(.i8));
        leader.yieldThen(.{});
    }
}

fn combineKernel(b: *B, cfg: CombineConfig) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .d = .{ .tensor = .{ .dtype = .f32, .shape = &.{ cfg.tokens * cfg.topk, cfg.hidden } } },
        .y = .{ .tensor = .{ .dtype = .bf16, .shape = &.{ cfg.tokens, cfg.hidden } } },
    });

    const token = b.blockIdx().y;
    const first_hidden = b.blockIdx().x.mul(cfg.columns_per_cta).add(b.threadIdx().x);
    for (0..@intCast(@divExact(cfg.columns_per_cta, 128))) |column| {
        const hidden = first_hidden.add(column * 128);
        var in_bounds = b.openIf(hidden.lt(cfg.hidden));
        var acc = b.cst(.f32, 0);
        for (0..@intCast(cfg.topk)) |slot| {
            acc = acc.add(a.d.get(.{ token.mul(cfg.topk).add(slot), hidden }).to(.f32));
        }
        a.y.set(.{ token, hidden }, acc.to(.bf16));
        in_bounds.yieldThen(.{});
    }
}

test "MXFP4 CuTe boundary kernels emit for decode and prefill batches" {
    inline for (.{ @as(i64, 1), 6, 96, 16_384 }) |tokens| {
        const quant_ir = try GroupedQuantize.emit(std.testing.allocator, .{
            .tokens = tokens,
            .hidden = 5120,
            .topk = 6,
            .group_size = 16,
            .capacity = @divTrunc(tokens * 6 + 15, 16) + 191,
        }, .{ 128, 1, 1 });
        defer std.testing.allocator.free(quant_ir);
        try std.testing.expect(std.mem.indexOf(u8, quant_ir, "griddepcontrol.launch_dependents;") != null);
        try std.testing.expect(std.mem.indexOf(u8, quant_ir, "cvt.rn.satfinite.e4m3x2.f32") != null);

        const combine_ir = try Combine.emit(std.testing.allocator, .{
            .tokens = tokens,
            .hidden = 5120,
            .topk = 6,
            .columns_per_cta = combineColumns(tokens, 5120),
        }, .{ 128, 1, 1 });
        defer std.testing.allocator.free(combine_ir);
        try std.testing.expect(std.mem.indexOf(u8, combine_ir, "scf.if") != null);
    }
}
