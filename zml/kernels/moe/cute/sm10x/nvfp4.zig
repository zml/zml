//! CuTe input quantization kernel in front of the persistent NVFP4 GEMMs of
//! `persistent_nvfp4.zig`: routed NVFP4 input quantization.
const std = @import("std");

const Nvfp4QuantizationStrategy = @import("zml/quantization").Nvfp4QuantizationStrategy;
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

pub const RowQuantConfig = struct {
    /// Input rows: tokens in front of the gate/up GEMM, routes in front of the down GEMM.
    rows: i64,
    /// Values per quantized row.
    width: i64,
    /// Values each input row holds, when fewer: the rest of the quantized row is zero,
    /// which pads the reduction dimension of the GEMM that reads it to whole K tiles.
    input_width: ?i64 = null,
    /// Physical rows each input row is written to.
    topk: i64,
    /// Rows stored per expert group.
    group_size: i64,
    capacity: i64,
    /// Route `r` goes to row `r * group_size`; `route_map` is not read.
    direct: bool = false,
    strategy: Nvfp4QuantizationStrategy,
    /// Input rows hold interleaved gate and up values, activated before their quantization.
    activation: ?CappedSwiglu = null,
    /// Input rows per entry of `pre_scale`, which multiplies them first; null leaves them as they are.
    pre_scale_rows: ?i64 = null,
    /// Rows are multiplied by their entry of `weights` after the activation.
    weights: bool = false,
};

const QuantizeRows = cute.Program(RowQuantConfig, .{
    .name = "nvfp4_cute_quantize_rows",
    .inputs = &.{ "x", "route_map", "ids", "pre_scale", "weights" },
    .outputs = &.{ "q", "s", "row_scale" },
    .run = quantizeRowsProgram,
});

const quantize_rows_kernel_name = "nvfp4_cute_quantize_rows_kernel";

/// NVFP4 quantization of activation rows with a dynamic scale per row, for experts
/// without calibrated activation scales. As `quantization.quantizeNvfp4Rows`, for each row:
///   1. `row_scale` is the power of two at or above `max|row| / 1536`, computed from the
///      row itself: the block scales of `row / row_scale` then fit E4M3;
///   2. every 16-value block of `row / row_scale` takes the E4M3 scale `cfg.strategy` picks;
///   3. values are rounded to E2M1 against the scale of their block.
/// The GEMM reads `q` and `s`, and its output for the row is then multiplied by its entry
/// of `row_scale`, `[rows]` FP32. Rows are scattered to the physical rows of their routes
/// like `groupedQuantize`.
pub fn quantizeRows(
    x: zml.Tensor,
    route_map: zml.Tensor,
    ids: zml.Tensor,
    pre_scale: ?zml.Tensor,
    weights: ?zml.Tensor,
    cfg: RowQuantConfig,
) struct { q: zml.Tensor, s: zml.Tensor, row_scale: zml.Tensor } {
    std.debug.assert(@rem(cfg.width, 16) == 0);
    std.debug.assert((pre_scale != null) == (cfg.pre_scale_rows != null) and (weights != null) == cfg.weights);
    const one = zml.Tensor.scalar(1.0, .f32);
    const result = QuantizeRows.call(
        .{ .x = x, .route_map = route_map, .ids = ids, .pre_scale = pre_scale orelse one, .weights = weights orelse one },
        .{
            .q = .init(.{ cfg.capacity * cfg.group_size, @divExact(cfg.width, 2) }, .u8),
            .s = .init(.{ cfg.capacity, @divExact(cfg.width, 16) * 128 }, .u8),
            .row_scale = .init(.{cfg.rows}, .f32),
        },
        .{ .cfg = cfg },
    );
    return .{ .q = result.q, .s = result.s, .row_scale = result.row_scale };
}

fn quantizeRowsProgram(b: *B, cfg: RowQuantConfig) cute.FinishError!void {
    const threads = rowQuantThreads(cfg);
    b.beginFunction(quantize_rows_kernel_name, .cuda_kernel);
    try quantizeRowsKernel(b, cfg);
    b.endFunction(.{ threads, 1, 1 });

    b.beginFunction(QuantizeRows.name, .host);
    const host = try b.declareArgs(.{
        .x = .{ .ptr = cute.DType.bf16 },
        .route_map = .{ .ptr = cute.DType.i32 },
        .ids = .{ .ptr = cute.DType.i32 },
        .pre_scale = .{ .ptr = cute.DType.f32 },
        .weights = .{ .ptr = cute.DType.f32 },
        .q = .{ .ptr = cute.DType.i8 },
        .s = .{ .ptr = cute.DType.i8 },
        .row_scale = .{ .ptr = cute.DType.f32 },
    });
    const one = b.constant(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.constant(.i32, @as(i32, @intCast(cfg.rows))), one, one },
        .block = .{ b.constant(.i32, threads), one, one },
        .dynamic_smem = b.kernelSmemSize(quantize_rows_kernel_name),
        .stream = b.cudaStream(),
        .cluster = .{ one, one, one },
        .use_pdl = true,
    });
    const launch = b.launchEx(quantize_rows_kernel_name, config, .{
        b.recastPointer(host.x, .i32, .gmem, 16), host.route_map,                           host.ids, host.pre_scale,
        host.weights,                             b.recastPointer(host.q, .i32, .gmem, 16), host.s,   host.row_scale,
    });
    b.returnHostStatus(b.cudaResultStatus(launch));
    b.endFunction(null);
}

/// Threads of the CTA that quantizes one row: one per packed word, in whole warps.
fn rowQuantThreads(cfg: RowQuantConfig) i32 {
    const words = @divExact(cfg.width, 8);
    std.debug.assert(words <= 1024);
    return @intCast(std.mem.alignForward(i64, words, 32));
}

/// One CTA per input row. Each thread holds eight consecutive values, so two lanes
/// cover one 16-value NVFP4 block; warps share their maxima through shared memory for
/// the row scale.
fn quantizeRowsKernel(b: *B, cfg: RowQuantConfig) cute.FinishError!void {
    const words = @divExact(cfg.width, 8);
    const input_words = @divExact(cfg.input_width orelse cfg.width, 8);
    const warps: usize = @intCast(@divExact(rowQuantThreads(cfg), 32));
    const routes = cfg.rows * cfg.topk;
    const a = try b.declareArgs(.{
        // Two BF16 values per word: a gate and its up value when rows are activated.
        .x = .{ .tensor = .{ .dtype = .i32, .shape = &.{ cfg.rows, if (cfg.activation != null) 8 * input_words else 4 * input_words } } },
        .route_map = .{ .tensor = .{ .dtype = .i32, .shape = &.{routes} } },
        .ids = .{ .tensor = .{ .dtype = .i32, .shape = &.{routes} } },
        .pre_scale = .{ .tensor = .{ .dtype = .f32, .shape = &.{if (cfg.pre_scale_rows) |per| @divExact(cfg.rows, per) else 1} } },
        .weights = .{ .tensor = .{ .dtype = .f32, .shape = &.{if (cfg.weights) cfg.rows else 1} } },
        // Eight FP4 values per word.
        .q = .{ .tensor = .{ .dtype = .i32, .shape = &.{ cfg.capacity * cfg.group_size, words } } },
        .s = .{ .tensor = .{ .dtype = .i8, .shape = &.{ cfg.capacity, @divExact(cfg.width, 16) * 128 } } },
        .row_scale = .{ .tensor = .{ .dtype = .f32, .shape = &.{cfg.rows} } },
    });
    const warp_maxima = b.makeTensor(b.allocSmem(.f32, @intCast(warps), 16), b.makeLayout(&.{@as(i64, @intCast(warps))}, .{}));

    const word = b.threadIdx().x;
    const lane = word.rem(32);
    // The routing and the previous projection write what this grid reads.
    b.waitForDependency();
    const row = b.blockIdx().x;
    const zero = b.constant(.f32, 0);
    const high_half: i32 = @bitCast(@as(u32, 0xffff0000));
    const pre_scale: ?V = if (cfg.pre_scale_rows) |per| a.pre_scale.get(.{row.div(per)}) else null;
    const weight: ?V = if (cfg.weights) a.weights.get(.{row}) else null;

    const in_range = word.lt(@as(i32, @intCast(words)));
    const has_input = word.lt(@as(i32, @intCast(input_words)));
    const input_word = b.select(has_input, word, b.constant(.i32, 0));
    var values: [8]V = undefined;
    if (cfg.activation) |caps| {
        for (&values, 0..) |*value, i| {
            const pair = a.x.get(.{ row, input_word.mul(8).add(@as(i32, @intCast(i))) });
            var gate = pair.shl(16).bitCast(.f32);
            var up = pair.bitAnd(high_half).bitCast(.f32);
            if (pre_scale) |factor| {
                gate = gate.mul(factor);
                up = up.mul(factor);
            }
            value.* = softcap(b, gate, caps.gate_cap).mul(sigmoid(b, gate)).mul(softcap(b, up, caps.up_cap));
        }
    } else for (0..4) |i| {
        const pair = a.x.get(.{ row, input_word.mul(4).add(@as(i32, @intCast(i))) });
        values[2 * i] = pair.shl(16).bitCast(.f32);
        values[2 * i + 1] = pair.bitAnd(high_half).bitCast(.f32);
        if (pre_scale) |factor| {
            values[2 * i] = values[2 * i].mul(factor);
            values[2 * i + 1] = values[2 * i + 1].mul(factor);
        }
    }
    var maximum = zero;
    for (&values) |*value| {
        if (weight) |w| value.* = value.*.mul(w);
        // Rows are rounded to BF16 like the inputs of BF16 GEMMs.
        if (cfg.activation != null or weight != null) value.* = value.*.to(.bf16).to(.f32);
        value.* = b.select(has_input, value.*, zero);
        maximum = maximum.maximum(value.*.abs());
    }

    // The row maximum: each warp's, then all of them through shared memory.
    const warp_maximum = b.warpMaxF32(maximum);
    var first_lane = b.openIf(lane.eq(0));
    warp_maxima.set(.{word.div(32)}, warp_maximum);
    first_lane.close();
    b.syncThreads();
    var row_maximum = zero;
    for (0..warps) |warp| row_maximum = row_maximum.maximum(warp_maxima.get(.{@as(i32, @intCast(warp))}));

    // A power of two keeps the block scales of the divided row in E4M3's normal range.
    const raw = row_maximum.mul(1.0 / 1536.0).maximum(std.math.floatMin(f32));
    const row_scale = raw.bitCast(.i32).add(0x7fffff).bitAnd(@as(i32, @bitCast(@as(u32, 0xff800000)))).bitCast(.f32);
    var first_thread = b.openIf(word.eq(0));
    a.row_scale.set(.{row}, row_scale);
    first_thread.close();

    var block_maximum = zero;
    for (&values) |*value| {
        value.* = value.*.div(row_scale);
        block_maximum = block_maximum.maximum(value.*.abs());
    }
    block_maximum = block_maximum.maximum(b.shuffleXor(block_maximum, 1));

    const six = blockCandidate(b, values, block_maximum, 6);
    const chosen: struct { word: V, bits: V } = switch (cfg.strategy) {
        .largest_maps_to_six => .{ .word = six.word, .bits = six.bits },
        .best_of_four_or_six => blk: {
            // The finer steps of amax/4 give up the largest value; ties keep amax/6.
            const four = blockCandidate(b, values, block_maximum, 4);
            const use_four = four.err.lt(six.err);
            break :blk .{ .word = b.select(use_four, four.word, six.word), .bits = b.select(use_four, four.bits, six.bits) };
        },
    };

    const safe_word = b.select(in_range, word, b.constant(.i32, 0));
    const block = safe_word.div(2);
    var in_range_scope = b.openIf(in_range);
    for (0..@intCast(cfg.topk)) |slot| {
        const route = row.mul(cfg.topk).add(slot);
        const physical = if (cfg.direct) route.mul(cfg.group_size) else a.route_map.get(.{route});
        // Routes of other expert-parallel ranks are not scheduled.
        var scheduled = b.openIf(physical.ge(0).bitAnd(a.ids.get(.{route}).ge(0)));
        a.q.set(.{ physical, safe_word }, chosen.word);
        var leader = b.openIf(lane.rem(2).eq(0));
        a.s.set(.{ physical.div(cfg.group_size), swizzledScaleOffset(physical.rem(cfg.group_size), block) }, chosen.bits.to(.i8));
        leader.close();
        scheduled.close();
    }
    in_range_scope.close();
    // The rows this CTA owns are complete: release the persistent GEMM.
    b.launchDependents();
}

fn sigmoid(b: *B, x: V) V {
    return b.constant(.f32, 1.0).div(x.mul(-1.0).exp().add(1.0));
}

/// `cap * tanh(x / cap)`, with the tanh written over the exponential.
fn softcap(b: *B, x: V, cap: f32) V {
    const tanh = b.constant(.f32, 1.0).sub(b.constant(.f32, 2.0).div(x.mul(2.0 / cap).exp().add(1.0)));
    return tanh.mul(cap);
}

const BlockCandidate = struct { word: V, bits: V, err: V };

/// The packed values of a lane's half block when the block's largest magnitude maps to
/// `value_max`, the block's E4M3 scale byte, and the block's absolute reconstruction error.
fn blockCandidate(b: *B, values: [8]V, block_maximum: V, value_max: f32) BlockCandidate {
    const clamped = block_maximum.mul(1.0 / value_max).maximum(0x1p-6).minimum(448.0);
    const zero = b.constant(.f32, 0);
    const bits = b.packFp8x4(clamped, zero, zero, zero).bitAnd(0xff);
    const scale = b.e4m3ToF32(bits);
    var quantized: [8]V = undefined;
    var err = zero;
    for (&quantized, values) |*q, value| {
        q.* = value.div(scale);
        err = err.add(roundE2m1(b, q.*).mul(scale).sub(value).abs());
    }
    return .{ .word = b.packFp4x8(quantized), .bits = bits, .err = err.add(b.shuffleXor(err, 1)) };
}

/// E2M1 round-to-nearest, ties-to-even, with saturation, as the packing conversion rounds.
fn roundE2m1(b: *B, x: V) V {
    const magnitude = x.abs();
    var rounded = b.constant(.f32, 6.0);
    const thresholds = [_]f32{ 5, 3.5, 2.5, 1.75, 1.25, 0.75, 0.25 };
    const values = [_]f32{ 4, 3, 2, 1.5, 1, 0.5, 0 };
    for (thresholds, values, 0..) |threshold, value, i| {
        // Even codes win midpoint ties.
        const below = if (i % 2 == 0) magnitude.le(threshold) else magnitude.lt(threshold);
        rounded = b.select(below, b.constant(.f32, value), rounded);
    }
    return b.select(x.lt(0.0), rounded.mul(-1.0), rounded);
}

test "NVFP4 CuTe row quantization emits with and without the activation" {
    inline for (.{ @as(i64, 1), 16 }) |tokens| {
        const plain = try QuantizeRows.emit(std.testing.allocator, .{
            .rows = tokens,
            .width = 2304,
            .topk = 8,
            .group_size = 16,
            .capacity = tokens * 8,
            .direct = tokens == 1,
            .strategy = .best_of_four_or_six,
        });
        defer std.testing.allocator.free(plain);
        try std.testing.expect(std.mem.indexOf(u8, plain, "cvt.rn.satfinite.e2m1x2.f32") != null);
        const activated = try QuantizeRows.emit(std.testing.allocator, .{
            .rows = tokens * 8,
            .width = 1152,
            .topk = 1,
            .group_size = 16,
            .capacity = tokens * 8,
            .direct = tokens == 1,
            .strategy = .best_of_four_or_six,
            .activation = .{ .gate_cap = 4, .up_cap = 25 },
            .pre_scale_rows = 8,
            .weights = true,
        });
        defer std.testing.allocator.free(activated);
        try std.testing.expect(std.mem.indexOf(u8, activated, "griddepcontrol.launch_dependents;") != null);
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
