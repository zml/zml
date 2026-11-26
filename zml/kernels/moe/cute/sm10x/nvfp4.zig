//! CuTe input kernels in front of the persistent NVFP4 GEMMs of `persistent_nvfp4.zig`:
//! routed NVFP4 input quantization, and the routing of rows `quantization.nvfp4.quantize` quantized.
const std = @import("std");

const stdx = @import("stdx");
const zml = @import("zml/core");
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
    const zero = b.constant(.f32, 0);
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
    const one = b.constant(.i32, 1);
    const words = @divExact(cfg.hidden, 8);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.constant(.i32, @as(i32, @intCast(@divTrunc(words + words_per_cta - 1, words_per_cta)))), b.constant(.i32, @as(i32, @intCast(cfg.tokens))), one },
        .block = .{ b.constant(.i32, words_per_cta), one, one },
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
    const safe_word = b.select(in_range, word, b.constant(.i32, 0));
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
        leader.close();
        scheduled.close();
    }
    in_range_scope.close();
    // The rows this CTA owns are complete: release the persistent GEMM.
    b.launchDependents();
}

/// The capped SwiGLU of interleaved gate and up values:
/// `cap_g * tanh(gate / cap_g) * sigmoid(gate) * cap_u * tanh(up / cap_u)`.
pub const CappedSwiglu = struct { gate_cap: f32, up_cap: f32 };

const RouteRowsConfig = struct {
    rows: i64,
    width: i64,
    padded_width: i64,
    topk: i64,
    group_size: i64,
    capacity: i64,
    direct: bool,
};

const RouteRows = cute.Program(RouteRowsConfig, .{
    .name = "nvfp4_cute_route_rows",
    .inputs = &.{ "values", "scales", "route_map", "ids" },
    .outputs = &.{ "q", "s" },
    .run = routeRowsProgram,
});

const route_rows_kernel_name = "nvfp4_cute_route_rows_kernel";

/// NVFP4 rows and their block scales, as `quantization.nvfp4.quantize` returns them for
/// `{.row, .k}` rows along `.k`, copied to the physical rows of their routes like
/// `groupedQuantize`: packed FP4 `[capacity * group_size, padded_width / 2]` and E4M3 scales
/// `[capacity, padded_width / 16 * 128]` in the 128x4 layout. `route_map` and the local
/// experts `ids` are `[rows * topk]`.
pub fn routeRows(in: struct { values: zml.Tensor, scales: zml.Tensor, route_map: zml.Tensor, ids: zml.Tensor }, opts: struct {
    /// Physical rows each input row is written to.
    topk: i64,
    /// Values per physical row: zeros past the row pad the reduction of the GEMM to whole K tiles.
    padded_width: i64,
    /// Rows stored per expert group.
    group_size: i64,
    capacity: i64,
    /// Route `r` goes to row `r * group_size`; `route_map` is not read.
    direct: bool,
}) struct { q: zml.Tensor, s: zml.Tensor } {
    const cfg: RouteRowsConfig = .{
        .rows = in.values.dim(.row),
        .width = in.values.dim(.k),
        .padded_width = opts.padded_width,
        .topk = opts.topk,
        .group_size = opts.group_size,
        .capacity = opts.capacity,
        .direct = opts.direct,
    };
    const routes = cfg.rows * cfg.topk;
    stdx.debug.assert(in.scales.dim(.row) == cfg.rows and in.scales.count() * 16 == in.values.count(), "routeRows expects an E4M3 scale per 16 values, got {f} and {f}", .{ in.values.shape(), in.scales.shape() });
    stdx.debug.assert(in.route_map.count() == routes and in.ids.count() == routes, "routeRows expects {} routes, got {f} and {f}", .{ routes, in.route_map.shape(), in.ids.shape() });
    // The 128x4 layout takes four blocks per row; a CTA has one thread per eight values, at most 1024.
    stdx.debug.assert(cfg.padded_width >= cfg.width and @rem(cfg.padded_width, 64) == 0 and cfg.padded_width <= 8 * 1024, "routeRows expects rows of {} values padded to a multiple of 64 up to 8192, got {}", .{ cfg.width, cfg.padded_width });
    const result = RouteRows.call(
        // Two E2M1 values per byte, the first in the low nibble, as the GEMMs read them.
        .{ .values = in.values.splitAxis(.k, .{ .k = .auto, .nibble = 2 }).bitCast(.u8), .scales = in.scales.bitCast(.u8), .route_map = in.route_map, .ids = in.ids },
        .{
            .q = .init(.{ cfg.capacity * cfg.group_size, @divExact(cfg.padded_width, 2) }, .u8),
            .s = .init(.{ cfg.capacity, @divExact(cfg.padded_width, 16) * 128 }, .u8),
        },
        .{ .cfg = cfg },
    );
    return .{ .q = result.q, .s = result.s };
}

fn routeRowsProgram(b: *B, cfg: RouteRowsConfig) cute.FinishError!void {
    // One thread per packed word of a physical row.
    const threads: i32 = @intCast(@divExact(cfg.padded_width, 8));
    b.beginFunction(route_rows_kernel_name, .cuda_kernel);
    try routeRowsKernel(b, cfg);
    b.endFunction(.{ threads, 1, 1 });

    b.beginFunction(RouteRows.name, .host);
    const host = try b.declareArgs(.{
        .values = .{ .ptr = cute.DType.i8 },
        .scales = .{ .ptr = cute.DType.i8 },
        .route_map = .{ .ptr = cute.DType.i32 },
        .ids = .{ .ptr = cute.DType.i32 },
        .q = .{ .ptr = cute.DType.i8 },
        .s = .{ .ptr = cute.DType.i8 },
    });
    const one = b.constant(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.constant(.i32, @as(i32, @intCast(cfg.rows))), one, one },
        .block = .{ b.constant(.i32, threads), one, one },
        .dynamic_smem = b.kernelSmemSize(route_rows_kernel_name),
        .stream = b.cudaStream(),
        .cluster = .{ one, one, one },
        .use_pdl = true,
    });
    const launch = b.launchEx(route_rows_kernel_name, config, .{
        b.recastPointer(host.values, .i32, .gmem, 16), host.scales, host.route_map, host.ids, b.recastPointer(host.q, .i32, .gmem, 16), host.s,
    });
    b.returnHostStatus(b.cudaResultStatus(launch));
    b.endFunction(null);
}

/// One CTA per input row. Each thread copies one packed word, eight values, so two
/// lanes cover one 16-value block, whose scale the even lane copies.
fn routeRowsKernel(b: *B, cfg: RouteRowsConfig) cute.FinishError!void {
    const words = @divExact(cfg.padded_width, 8);
    const input_words = @divExact(cfg.width, 8);
    const routes = cfg.rows * cfg.topk;
    const a = try b.declareArgs(.{
        // Eight FP4 values per word.
        .values = .{ .tensor = .{ .dtype = .i32, .shape = &.{ cfg.rows, input_words } } },
        .scales = .{ .tensor = .{ .dtype = .i8, .shape = &.{ cfg.rows, @divExact(cfg.width, 16) } } },
        .route_map = .{ .tensor = .{ .dtype = .i32, .shape = &.{routes} } },
        .ids = .{ .tensor = .{ .dtype = .i32, .shape = &.{routes} } },
        .q = .{ .tensor = .{ .dtype = .i32, .shape = &.{ cfg.capacity * cfg.group_size, words } } },
        .s = .{ .tensor = .{ .dtype = .i8, .shape = &.{ cfg.capacity, @divExact(cfg.padded_width, 16) * 128 } } },
    });

    const word = b.threadIdx().x;
    // The routing and the quantizer write what this grid reads.
    b.waitForDependency();
    const row = b.blockIdx().x;
    const has_input = word.lt(@as(i32, @intCast(input_words)));
    const input_word = b.select(has_input, word, b.constant(.i32, 0));
    const value = b.select(has_input, a.values.get(.{ row, input_word }), 0);
    const scale = b.select(has_input, a.scales.get(.{ row, input_word.div(2) }), 0);
    const block = word.div(2);
    for (0..@intCast(cfg.topk)) |slot| {
        const route = row.mul(cfg.topk).add(slot);
        const physical = if (cfg.direct) route.mul(cfg.group_size) else a.route_map.get(.{route});
        // Routes of other expert-parallel ranks are not scheduled.
        var scheduled = b.openIf(physical.ge(0).bitAnd(a.ids.get(.{route}).ge(0)));
        a.q.set(.{ physical, word }, value);
        var leader = b.openIf(word.rem(2).eq(0));
        a.s.set(.{ physical.div(cfg.group_size), swizzledScaleOffset(physical.rem(cfg.group_size), block) }, scale);
        leader.close();
        scheduled.close();
    }
    // The rows this CTA owns are complete: release the persistent GEMM.
    b.launchDependents();
}

test "NVFP4 CuTe row routing emits in front of both GEMMs" {
    for ([_]i64{ 1, 16 }) |tokens| {
        const configs = [_]RouteRowsConfig{
            // Tokens in front of the gate/up GEMM, then routes in front of the down GEMM.
            .{ .rows = tokens, .width = 2304, .padded_width = 2304, .topk = 8, .group_size = 16, .capacity = tokens * 8, .direct = tokens == 1 },
            .{ .rows = tokens * 8, .width = 1152, .padded_width = 1280, .topk = 1, .group_size = 16, .capacity = tokens * 8, .direct = tokens == 1 },
        };
        for (configs) |cfg| {
            const ir = try RouteRows.emit(std.testing.allocator, cfg);
            defer std.testing.allocator.free(ir);
            try std.testing.expect(std.mem.indexOf(u8, ir, "griddepcontrol.launch_dependents;") != null);
            try std.testing.expect(std.mem.indexOf(u8, ir, "cuda.launch_ex") != null);
        }
    }
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
