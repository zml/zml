//! CuTe input quantization kernel in front of the persistent NVFP4 GEMMs of
//! `persistent_nvfp4.zig`: routed NVFP4 input quantization.
const std = @import("std");

const zml = @import("../../../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;

pub const GroupedQuantConfig = struct {
    tokens: i64,
    hidden: i64,
    topk: i64,
    experts: i64,
    /// Rows stored per expert group.
    group_size: i64,
    capacity: i64,
    /// Route `r` goes to row `r * group_size`; `route_map` is not read.
    direct: bool = false,
};

const GroupedQuantize = cute.Program(GroupedQuantConfig, .{
    .name = "nvfp4_cute_grouped_quantize",
    .inputs = &.{ "x", "route_map", "ids", "global_scale" },
    .outputs = &.{ "q", "s" },
    .run = groupedQuantizeProgram,
});

const quantize_kernel_name = "nvfp4_cute_grouped_quantize_kernel";
/// Packed FP4 words (eight values) a CTA quantizes per token.
const words_per_cta = 128;

/// NVFP4 E4M3 block scale of a block maximum: `scale = e4m3(global *
/// maximum / 6)` and the quantization multiplier `global / scale`, zero for an
/// all-zero block.
pub fn blockScale(b: *B, maximum: V, global: V) struct { bits: V, multiplier: V } {
    const raw = global.mul(maximum.mul(1.0 / 6.0)).minimum(448.0);
    const zero = b.cst(.f32, 0);
    const bits = b.packFp8x4(raw, zero, zero, zero).bitAnd(0xff);
    const scale = b.e4m3ToF32(bits);
    const inverse = b.unaryF32("rcp.approx.ftz.f32", scale.mul(b.unaryF32("rcp.approx.ftz.f32", global)));
    return .{ .bits = bits, .multiplier = b.select(scale.ne(0.0), inverse, zero) };
}

/// Offset of the scale of row `row` (within its 128-row group) and 16-wide
/// block `block` in the 128x4 tensor-core layout.
pub fn swizzledScaleOffset(row: V, block: V) V {
    return block.div(4).mul(512)
        .add(row.rem(32).mul(16))
        .add(row.div(32).mul(4))
        .add(block.rem(4));
}

/// Quantize each token once per routed expert and scatter it to the physical
/// rows assigned by the routing: packed FP4 `[capacity * group, hidden / 2]`
/// and E4M3 scales `[capacity, hidden / 16 * 128]` in the 128x4 layout.
/// `ids` are the `[tokens * topk]` local experts and `global_scale` the
/// `[experts]` FP32 activation global scales.
pub fn groupedQuantize(x: zml.Tensor, route_map: zml.Tensor, ids: zml.Tensor, global_scale: zml.Tensor, cfg: GroupedQuantConfig) struct { q: zml.Tensor, s: zml.Tensor } {
    std.debug.assert(@rem(cfg.hidden, 64) == 0);
    const result = GroupedQuantize.call(
        .{ .x = x, .route_map = route_map, .ids = ids, .global_scale = global_scale },
        .{
            .q = .init(.{ cfg.capacity * cfg.group_size, @divExact(cfg.hidden, 2) }, .u8),
            .s = .init(.{ cfg.capacity, @divExact(cfg.hidden, 16) * 128 }, .u8),
        },
        .{ .cfg = cfg },
    );
    return .{ .q = result.q, .s = result.s };
}

fn groupedQuantizeProgram(b: *B, cfg: GroupedQuantConfig) cute.FinishError!void {
    b.beginFunction(quantize_kernel_name, .cuda_kernel);
    try groupedQuantizeKernel(b, cfg);
    b.endFunction(.{ words_per_cta, 1, 1 });

    b.beginFunction(GroupedQuantize.name, .host);
    const host = try b.declareArgs(.{
        .x = .{ .ptr = cute.DType.bf16 },
        .route_map = .{ .ptr = cute.DType.i32 },
        .ids = .{ .ptr = cute.DType.i32 },
        .global_scale = .{ .ptr = cute.DType.f32 },
        .q = .{ .ptr = cute.DType.i8 },
        .s = .{ .ptr = cute.DType.i8 },
    });
    const one = b.cst(.i32, 1);
    const words = @divExact(cfg.hidden, 8);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.cst(.i32, @as(i32, @intCast(@divTrunc(words + words_per_cta - 1, words_per_cta)))), b.cst(.i32, @as(i32, @intCast(cfg.tokens))), one },
        .block = .{ b.cst(.i32, words_per_cta), one, one },
        .dynamic_smem = b.kernelSmemSize(quantize_kernel_name),
        .stream = b.cudaStream(),
        .cluster = .{ one, one, one },
        .use_pdl = true,
    });
    const launch = b.launchEx(quantize_kernel_name, config, .{
        b.recastPointer(host.x, .i32, .gmem, 16), host.route_map, host.ids, host.global_scale, b.recastPointer(host.q, .i32, .gmem, 16), host.s,
    });
    b.returnHostStatus(b.cudaResultStatus(launch));
    b.endFunction(null);
}

/// Each thread quantizes eight consecutive values into one packed word, so
/// two lanes cover one 16-value NVFP4 block.
fn groupedQuantizeKernel(b: *B, cfg: GroupedQuantConfig) cute.FinishError!void {
    const words = @divExact(cfg.hidden, 8);
    const routes = cfg.tokens * cfg.topk;
    const a = try b.declareArgs(.{
        // Two BF16 values per word.
        .x = .{ .tensor = .{ .dtype = .i32, .shape = &.{ cfg.tokens, @divExact(cfg.hidden, 2) } } },
        .route_map = .{ .tensor = .{ .dtype = .i32, .shape = &.{routes} } },
        .ids = .{ .tensor = .{ .dtype = .i32, .shape = &.{routes} } },
        .global_scale = .{ .tensor = .{ .dtype = .f32, .shape = &.{cfg.experts} } },
        // Eight FP4 values per word.
        .q = .{ .tensor = .{ .dtype = .i32, .shape = &.{ cfg.capacity * cfg.group_size, words } } },
        .s = .{ .tensor = .{ .dtype = .i8, .shape = &.{ cfg.capacity, @divExact(cfg.hidden, 16) * 128 } } },
    });

    const tid = b.threadIdx().x;
    // The routing and the previous layer write what this grid reads.
    b.waitForDependency();
    const token = b.blockIdx().y;
    const word = b.blockIdx().x.mul(words_per_cta).add(tid);
    // Out-of-range lanes still take part in the pair reduction.
    const in_range = word.lt(@as(i32, @intCast(words)));
    const safe_word = b.select(in_range, word, b.cst(.i32, 0));
    var x: [8]V = undefined;
    for (0..4) |i| {
        const pair = a.x.get(.{ token, safe_word.mul(4).add(@as(i32, @intCast(i))) });
        x[2 * i] = pair.shl(16).bitCast(.f32);
        x[2 * i + 1] = pair.bitAnd(@as(i32, @bitCast(@as(u32, 0xffff0000)))).bitCast(.f32);
    }
    var maximum = x[0].abs();
    for (x[1..]) |value| maximum = maximum.maximum(value.abs());
    maximum = maximum.maximum(b.shuffleXor(maximum, 1));

    const block = safe_word.div(2);
    var in_range_scope = b.openIf(in_range);
    for (0..@intCast(cfg.topk)) |slot| {
        const route = token.mul(cfg.topk).add(slot);
        const row = if (cfg.direct) route.mul(cfg.group_size) else a.route_map.get(.{route});
        // Routes of other expert-parallel ranks are not scheduled.
        const expert = a.ids.get(.{route});
        var scheduled = b.openIf(row.ge(0).bitAnd(expert.ge(0)));
        const scale = blockScale(b, maximum, a.global_scale.get(.{expert}));
        var quantized: [8]V = undefined;
        for (&quantized, x) |*q, value| q.* = value.mul(scale.multiplier);
        a.q.set(.{ row, safe_word }, b.packFp4x8(quantized));

        var leader = b.openIf(tid.rem(2).eq(0));
        a.s.set(.{ row.div(cfg.group_size), swizzledScaleOffset(row.rem(cfg.group_size), block) }, scale.bits.to(.i8));
        leader.yieldThen(.{});
        scheduled.yieldThen(.{});
    }
    in_range_scope.yieldThen(.{});
    // The rows this CTA owns are complete: release the persistent GEMM.
    b.launchDependents();
}

test "NVFP4 CuTe quantization emits for decode and prefill batches" {
    inline for (.{ @as(i64, 1), 6, 96, 16_384 }) |tokens| {
        const ir = try GroupedQuantize.emit(std.testing.allocator, .{
            .tokens = tokens,
            .hidden = 5120,
            .topk = 6,
            .experts = 192,
            .group_size = 16,
            .capacity = @divTrunc(tokens * 6 + 15, 16) + 191,
            .direct = tokens == 1,
        });
        defer std.testing.allocator.free(ir);
        try std.testing.expect(std.mem.indexOf(u8, ir, "griddepcontrol.launch_dependents;") != null);
        try std.testing.expect(std.mem.indexOf(u8, ir, "cuda.launch_ex") != null);
        try std.testing.expect(std.mem.indexOf(u8, ir, "cvt.rn.satfinite.e2m1x2.f32") != null);
    }
}
