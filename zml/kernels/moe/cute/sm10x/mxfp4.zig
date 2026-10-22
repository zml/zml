//! CuTe input quantization kernel in front of the persistent MXFP4 GEMMs of
//! `persistent_mxfp4.zig`: routed MXFP8 input quantization.
const std = @import("std");

const zml = @import("../../../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const launchBoundary = @import("common.zig").launchBoundary;

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

// A program rather than a bare kernel so its launch carries the
// programmatic-dependent-launch attribute: it then starts while the previous
// kernel drains, and waits on `griddepcontrol` before its first read. A bare
// kernel is launched by XLA without that attribute.
const GroupedQuantize = cute.Program(GroupedQuantConfig, .{
    .name = "mxfp4_cute_grouped_quantize",
    .inputs = &.{ "x", "route_map" },
    .outputs = &.{ "q", "s" },
    .run = groupedQuantizeProgram,
});

const quantize_kernel_name = "mxfp4_cute_grouped_quantize_kernel";

/// Quantize each token once, then scatter it to the physical rows assigned by
/// the routing. Scale bytes use the 128x4 layout consumed by the SM100
/// block-scaled MMA scale-factor tensor map.
pub fn groupedQuantize(x: zml.Tensor, route_map: zml.Tensor, cfg: GroupedQuantConfig) struct { q: zml.Tensor, s: zml.Tensor } {
    std.debug.assert(cfg.group_size == 8 or cfg.group_size == 16 or cfg.group_size == 32 or cfg.group_size == 64 or cfg.group_size == 128);
    const result = GroupedQuantize.call(
        .{ .x = x, .route_map = route_map },
        .{
            .q = .init(.{ cfg.capacity * cfg.group_size, cfg.hidden }, .f8e4m3fn),
            .s = .init(.{ cfg.capacity, @divExact(cfg.hidden, 32) * 128 }, .u8),
        },
        .{ .cfg = cfg },
    );
    return .{ .q = result.q, .s = result.s };
}

/// Python `prepare_input_vector`: each thread quantizes four consecutive
/// values and stores them as one packed word, so eight lanes cover one
/// 32-value MX block.
fn groupedQuantizeProgram(b: *B, cfg: GroupedQuantConfig) cute.FinishError!void {
    b.beginFunction(quantize_kernel_name, .cuda_kernel);
    try groupedQuantizeKernel(b, cfg);
    b.endFunction(.{ 128, 1, 1 });

    b.beginFunction(GroupedQuantize.name, .host);
    // Host parameters carry the custom call's element types; the kernel reads
    // four FP8 values per 32-bit word.
    const host = try b.declareArgs(.{
        .x = .{ .ptr = cute.DType.bf16 },
        .route_map = .{ .ptr = cute.DType.i32 },
        .q = .{ .ptr = cute.DType.f8e4m3fn },
        .s = .{ .ptr = cute.DType.i8 },
    });
    try launchBoundary(b, quantize_kernel_name, .{ @intCast(@divExact(cfg.hidden, 4 * 128)), @intCast(cfg.tokens), 1 }, .{
        host.x, host.route_map, b.recastPointer(host.q, .i32, .gmem, 16), host.s,
    });
    b.endFunction(null);
}

fn groupedQuantizeKernel(b: *B, cfg: GroupedQuantConfig) cute.FinishError!void {
    std.debug.assert(@rem(cfg.hidden, 4 * 128) == 0);
    const a = try b.declareArgs(.{
        .x = .{ .tensor = .{ .dtype = .bf16, .shape = &.{ cfg.tokens, cfg.hidden } } },
        .route_map = .{ .tensor = .{ .dtype = .i32, .shape = &.{cfg.tokens * cfg.topk} } },
        // Four FP8 values per word.
        .q = .{ .tensor = .{ .dtype = .i32, .shape = &.{ cfg.capacity * cfg.group_size, @divExact(cfg.hidden, 4) } } },
        .s = .{ .tensor = .{ .dtype = .i8, .shape = &.{ cfg.capacity, @divExact(cfg.hidden, 32) * 128 } } },
    });

    const tid = b.threadIdx().x;
    // This grid may have started before its producer finished: the routing and
    // the previous layer write what it reads below.
    b.waitForDependency();
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
        // Routes of other expert-parallel ranks are not scheduled.
        var scheduled = if (cfg.direct) null else b.openIf(row.ge(0));
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
        if (scheduled) |*s| s.yieldThen(.{});
    }
    // Only now are the rows this CTA owns complete: release the persistent
    // GEMM, which waits on this signal before its first TMA read.
    b.launchDependents();
}

test "MXFP4 CuTe input quantization emits for decode and prefill batches" {
    inline for (.{ @as(i64, 1), 6, 96, 16_384 }) |tokens| {
        const quant_ir = try GroupedQuantize.emit(std.testing.allocator, .{
            .tokens = tokens,
            .hidden = 5120,
            .topk = 6,
            .group_size = 16,
            .capacity = @divTrunc(tokens * 6 + 15, 16) + 191,
        });
        defer std.testing.allocator.free(quant_ir);
        try std.testing.expect(std.mem.indexOf(u8, quant_ir, "griddepcontrol.launch_dependents;") != null);
        try std.testing.expect(std.mem.indexOf(u8, quant_ir, "cuda.launch_ex") != null);
        try std.testing.expect(std.mem.indexOf(u8, quant_ir, "cvt.rn.satfinite.e4m3x2.f32") != null);
    }
}
