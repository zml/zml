//! NVFP4 activation quantization.
const std = @import("std");

const stdx = @import("stdx");
const zml = @import("zml/core");
const DataType = zml.DataType;
const Tensor = zml.Tensor;

const quantization = @import("quantization.zig");
const QuantizedInput = quantization.QuantizedInput;

pub const block_size = 16;

/// Quantize activations on CUDA with compute capability 10.0 or newer, the way the weights
/// they multiply ask for in `Quantization.nvfp4_input`.
pub fn quantizeInput(input: Tensor, axis: zml.Shape.Tag, strategy: ?QuantizationStrategy, platform: *const zml.Platform) ?QuantizedInput {
    const capability = zml.platform.cuda.computeCapability(platform) orelse return null;
    if (!capability.atLeast(.{ .major = 10, .minor = 0 })) return null;

    return quantize(input, axis, strategy orelse @panic("NVFP4 weights need Quantization.nvfp4_input to quantize their activations"));
}

/// How NVFP4 activations are scaled: a global scale on top of the E4M3 scale of each block.
/// NVFP4 weights carry the strategy of the activations they multiply in
/// `Quantization.nvfp4_input`, set when the checkpoint is read.
pub const QuantizationStrategy = struct {
    global_scale: union(enum) {
        /// Block scales cover the values as they are.
        none,
        /// The calibrated scale of the whole tensor.
        calibrated: quantization.Quantization.GlobalScale,
        /// Each row along the quantized axis takes its own `amax / 1536` ("per_token").
        /// This lets block scales reach 256 with amax/6 and 384 with amax/4, within E4M3.
        per_row,
    },
    block_scale: BlockScale,

    /// How NVFP4 quantization picks the E4M3 scale of each 16-value block.
    pub const BlockScale = enum {
        /// The block's largest magnitude maps to 6, the largest E2M1 value.
        largest_maps_to_six,
        /// Whichever of amax/6 and amax/4 reconstructs the block with the lower absolute error
        /// ("fouroversix_mae"). amax/4 gives up the largest value for finer steps. Ties keep amax/6.
        best_of_four_or_six,
    };
};

/// NVFP4 quantization along `axis` of `x` read as BF16, the precision of activations.
/// The result's global scale is null with `.none`, the calibrated multiplier with `.calibrated`
/// and, with `.per_row`, one FP32 scale per row: `x`'s shape without `axis`.
///
/// `.per_row` with `.best_of_four_or_six` is the quantization of vLLM's Inkling-2 experts,
/// whose `nvfp4_quantize` this follows operation by operation. The two agree bit for bit on
/// normal values where FP32 division is correctly rounded. XLA's CUDA backend divides with
/// `div.full.f32` unless it runs with `-nvptx-prec-divf32=2`: values that land exactly on an
/// E2M1 rounding tie, about one in 12000, then take the neighbouring code.
pub fn quantize(x: Tensor, axis: anytype, strategy: QuantizationStrategy) QuantizedInput {
    stdx.debug.assert(x.shape().hasTag(axis) != null, "NVFP4 quantization expects x to have {any} tag, got {f}", .{ axis, x.shape() });
    stdx.debug.assert(@rem(x.dim(axis), block_size) == 0, "NVFP4 quantization expects {any} to be a multiple of {}, got {f}", .{ axis, block_size, x.shape() });

    // XLA drops a conversion to BF16 and back; reducing the precision keeps the rounding.
    const blocks = x.convert(.f32).reducePrecision(8, 7).splitAxis(axis, .{ .sc = -1, .blk = block_size });
    const block_max = blocks.abs().max(.blk);
    const one: Tensor = .scalar(1, .f32);
    const scaling: Scaling = switch (strategy.global_scale) {
        .none => .{ .encoder = one, .decoder = one, .outer = one },
        .calibrated => |scale| blk: {
            const outer = scale.asMultiplier();
            stdx.debug.assert(outer.rank() == 0, "NVFP4 quantization expects one calibrated scale for the tensor, got {f}", .{scale.value.shape()});
            break :blk .{ .encoder = one.div(outer), .decoder = outer, .outer = outer };
        },
        .per_row => blk: {
            const amax = block_max.max(.sc);
            // A row of zeros has a capped encoder and quantizes to zeros.
            const encoder = Tensor.scalar(1536, .f32).div(amax).minimum(.scalar(std.math.floatMax(f32), .f32));
            break :blk .{ .encoder = encoder, .decoder = one.div(encoder), .outer = amax.scale(1.0 / 1536.0) };
        },
    };

    const six = blockScale(block_max, scaling.encoder, 6);
    const scale = switch (strategy.block_scale) {
        .largest_maps_to_six => six,
        .best_of_four_or_six => blk: {
            const four = blockScale(block_max, scaling.encoder, 4);
            break :blk reconstructionError(blocks, four, scaling).cmp(.LT, reconstructionError(blocks, six, scaling)).select(four, six);
        },
    };

    return .{
        .values = quantizeValues(blocks, scale, scaling.decoder).reshape(x.shape().withDtype(.f4e2m1)),
        .scales = scale.convert(.f8e4m3fn).squeeze(.blk),
        .global_scale = switch (strategy.global_scale) {
            .none => null,
            .calibrated => scaling.outer,
            .per_row => scaling.outer.squeeze(.blk).squeeze(.sc),
        },
    };
}

/// What `quantize` rounded its input to, in FP32: the values of `input` along `axis` times their
/// block scales and its global scale. Without a global scale BF16 holds the result exactly: a
/// GEMM can read it and leave the global scale to its output.
pub fn dequantize(input: QuantizedInput, axis: anytype) Tensor {
    const blocks = input.values.convert(.f32).splitAxis(axis, .{ .sc = -1, .blk = block_size });
    const rows = blocks.mul(input.scales.convert(.f32).broad(blocks.shape())).reshape(input.values.shape().withDtype(.f32));
    const global_scale = input.global_scale orelse return rows;
    return rows.mul(global_scale.broad(rows.shape()));
}

/// A global scale in the three forms the quantization reads, which vLLM computes separately
/// for a row; each one is a scalar or broadcasts to the blocks.
const Scaling = struct {
    /// 1 / global scale: multiplies block maxima before their block scale is taken.
    encoder: Tensor,
    /// 1 / encoder: multiplies block scales to give what the values are divided by.
    decoder: Tensor,
    /// The global scale itself, which dequantizes the blocks.
    outer: Tensor,
};

/// A conversion to E4M3 or E2M1 that a conversion back to FP32 cannot cancel.
/// XLA keeps FP32 round trips through floats of 8 bits or fewer, so they need no barrier.
/// HACK: the oneAPI plugin's XLA predates this and drops them. The barrier keeps them there,
/// and ends the fusion; it goes away with a oneAPI plugin whose XLA keeps these round trips.
/// It misses "preserve FP8 round trips" change (286cef081b).
fn keepConversion(converted: Tensor) Tensor {
    return if (zml.Compiler.current().platform.target == .oneapi) converted.optimizationBarrier() else converted;
}

/// The E4M3 scale, in FP32, of blocks whose largest magnitude maps to `value_max`.
fn blockScale(block_max: Tensor, encoder: Tensor, comptime value_max: f32) Tensor {
    const scale = block_max.mul(encoder.scale(1.0 / value_max).broad(block_max.shape()))
        .minimum(.scalar(DataType.f8e4m3fn.maxValue().as(f32), .f32));
    return keepConversion(scale.convert(.f8e4m3fn)).convert(.f32);
}

/// The E2M1 values of `blocks` under the block scales `scale`. They are what `quantize`
/// returns: a caller converting them back to FP32 reads rounded values too.
fn quantizeValues(blocks: Tensor, scale: Tensor, decoder: Tensor) Tensor {
    // A block too small for E4M3 has a zero scale; its capped inverse saturates values that
    // the scale then reads as zeros.
    const inverse = Tensor.scalar(1, .f32).div(scale.mul(decoder.broad(scale.shape())))
        .minimum(.scalar(std.math.floatMax(f32), .f32));
    return keepConversion(blocks.mul(inverse.broad(blocks.shape())).convert(.f4e2m1));
}

/// Each block's absolute reconstruction error under the block scales `scale`.
fn reconstructionError(blocks: Tensor, scale: Tensor, scaling: Scaling) Tensor {
    var sum = quantizeValues(blocks, scale, scaling.decoder).convert(.f32)
        .mul(scale.broad(blocks.shape()))
        .mul(scaling.outer.broad(blocks.shape()))
        .sub(blocks).abs();
    // Adjacent pairs first, like vLLM: a reduction's order depends on the backend, and the
    // order decides between two scales whose errors are an ulp apart.
    inline for (.{ 8, 4, 2, 1 }) |pairs| {
        const split = sum.splitAxis(.blk, .{ .blk = pairs, .pair = 2 });
        sum = split.slice(.pair, .single(0)).add(split.slice(.pair, .single(1)));
    }
    return sum;
}

test "NVFP4 four-or-six keeps the block scale with the lower error, a calibrated scale is factored out" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const Local = struct {
        const Outputs = struct { six: Tensor, best: Tensor, scales: Tensor, calibrated: Tensor, calibrated_scale: Tensor };

        fn forward(x: Tensor) Outputs {
            const best = quantize(x, .k, .{ .global_scale = .none, .block_scale = .best_of_four_or_six });
            const six = quantize(x, .k, .{ .global_scale = .none, .block_scale = .largest_maps_to_six });
            // Under a global scale of 1/2, x/2 has the values and block scales of x.
            const calibrated = quantize(x.scale(0.5), .k, .{
                .global_scale = .{ .calibrated = .{ .value = .scalar(2, .f32), .operation = .divide } },
                .block_scale = .best_of_four_or_six,
            });
            return .{
                .six = dequantize(six, .k),
                .best = dequantize(best, .k),
                .scales = best.scales.convert(.f32),
                .calibrated = dequantize(.{ .values = calibrated.values, .scales = calibrated.scales, .global_scale = null }, .k),
                .calibrated_scale = calibrated.global_scale.?,
            };
        }
    };
    const x: Tensor = .init(.{ .k = 2 * block_size }, .f32);
    var exe = try platform.compileFn(allocator, io, Local.forward, .{x}, .{});
    defer exe.deinit();
    // The first block peaks at 4 and is exact with amax/4; the second peaks at 6 and is exact with amax/6.
    const pattern = [2][8]f32{ .{ 4, 3, 2, 1.5, 1, 0.5, 0, -4 }, .{ 6, 3, 1.5, 0.5, 1, -6, 2, 0 } };
    var host: [2 * block_size]f32 = undefined;
    for (&host, 0..) |*value, i| value.* = pattern[i / block_size][i % 8];
    var buffer = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&host));
    defer buffer.deinit();
    var output: zml.Bufferized(Local.Outputs) = undefined;
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    runner.run(io, .{buffer}, .{&output}, .{ .wait = true });
    defer zml.Buffer.deinitAll(Local.Outputs, &output);
    var six = try output.six.toSliceAlloc(allocator, io);
    defer six.free(allocator);
    var best = try output.best.toSliceAlloc(allocator, io);
    defer best.free(allocator);
    var scales = try output.scales.toSliceAlloc(allocator, io);
    defer scales.free(allocator);
    var calibrated = try output.calibrated.toSliceAlloc(allocator, io);
    defer calibrated.free(allocator);
    var calibrated_scale = try output.calibrated_scale.toSliceAlloc(allocator, io);
    defer calibrated_scale.free(allocator);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1 }, scales.constItems(f32));
    try std.testing.expectEqualSlices(f32, &host, best.constItems(f32));
    try std.testing.expectEqualSlices(f32, &host, calibrated.constItems(f32));
    try std.testing.expectEqualSlices(f32, &.{0.5}, calibrated_scale.constItems(f32));
    var six_error: f32 = 0;
    for (host[0..block_size], six.constItems(f32)[0..block_size]) |expected, actual| six_error += @abs(expected - actual);
    try std.testing.expect(six_error > 0);
    try std.testing.expectEqualSlices(f32, host[block_size..], six.constItems(f32)[block_size..]);
}

test "FP32 round trips through E2M1 and E4M3 round to nearest even inside a graph" {
    // quantize relies on both conversions being kept, with ties to even and E2M1 saturation.
    const Local = struct {
        const Outputs = struct { e2m1: Tensor, e4m3: Tensor };

        fn forward(e2m1: Tensor, e4m3: Tensor) Outputs {
            return .{
                .e2m1 = keepConversion(e2m1.convert(.f4e2m1)).convert(.f32),
                .e4m3 = keepConversion(e4m3.convert(.f8e4m3fn)).convert(.f32),
            };
        }
    };
    const e2m1 = [_]f32{ -7, -5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7, std.math.inf(f32) };
    const e2m1_rounded = [e2m1.len]f32{ -6, -4, -4, -2, -2, -1, -1, 0, 0, 0, 1, 1, 2, 2, 4, 4, 6, 6 };
    // Midpoints of normal and subnormal values, and both ends of the range.
    const e4m3 = [_]f32{ 1.0625, 1.1875, 0.3, 448, 0x1p-9, 0x1p-10, 0x3p-10, 0 };
    const e4m3_rounded = [e4m3.len]f32{ 1, 1.25, 0.3125, 448, 0x1p-9, 0, 0x1p-8, 0 };

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var exe = try platform.compileFn(allocator, io, Local.forward, .{ Tensor.init(.{e2m1.len}, .f32), Tensor.init(.{e4m3.len}, .f32) }, .{});
    defer exe.deinit();
    var e2m1_buffer = try zml.Buffer.fromBytes(io, platform, .init(.{e2m1.len}, .f32), std.mem.asBytes(&e2m1));
    defer e2m1_buffer.deinit();
    var e4m3_buffer = try zml.Buffer.fromBytes(io, platform, .init(.{e4m3.len}, .f32), std.mem.asBytes(&e4m3));
    defer e4m3_buffer.deinit();
    var output: zml.Bufferized(Local.Outputs) = undefined;
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    runner.run(io, .{ e2m1_buffer, e4m3_buffer }, .{&output}, .{ .wait = true });
    defer zml.Buffer.deinitAll(Local.Outputs, &output);
    var actual_e2m1 = try output.e2m1.toSliceAlloc(allocator, io);
    defer actual_e2m1.free(allocator);
    var actual_e4m3 = try output.e4m3.toSliceAlloc(allocator, io);
    defer actual_e4m3.free(allocator);
    try std.testing.expectEqualSlices(f32, &e2m1_rounded, actual_e2m1.constItems(f32));
    try std.testing.expectEqualSlices(f32, &e4m3_rounded, actual_e4m3.constItems(f32));
}

test "NVFP4 row quantization matches vLLM's kernel" {
    // What `nvfp4_quantize` of vLLM (vllm/models/inkling2/nvidia/ops/nvfp4.py at ed686ccec5) returns
    // on a GB300 for the rows below: E2M1 codes with the sign in bit 3, block scales, row maxima.
    const rows = 8;
    const k = 128;
    const codes = [rows]*const [k]u8{
        "0e890a81c083802d89790105a9e182182ac80801801816003c2b0817fdaa5018be58e05fb8f80823001e0b1881a0f9a380481dc27100a0808940600c102f13c9",
        "88000b97000cf0fb7000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "98d8e208880498830000000000000000777ffff77f77777fb90128060b0a840180840c00b3bff18828b0111170008888f08b7239e88e9900618a8090a00291f0",
        "611a3456ac534dd6699a3c562c5b455e79224d67bdec56ef6811ab452c5b4d5679aacd6fb5ec56eff1aa4567b5e4de6f71aac5e7bd645667e91a3c5ea4db45d6",
        "f122c567b5ecd66fe01aa446acdbc5de7922cd6f35ecdee7f9aac56fb5ec5ee7601a244eac53cd5e6012a44e2c534dd6f9aa4d67b5ecd66f689a2cce24db45d6",
        "6192bc5eacd3c5def9aa4defbde4d6ef699abc56ac5345d6f1924d6f3d64de6f71a24d6f356c5e6ff1aa45e7b5e4d67f7192cd6f3564de6fe19abcd6acd3cd5e",
        "e19ab4dea453cd5e79224567b564de6f61123456a45345def92acde7b56cde6fe91ab4562c5345de691abcde2cd34ddef9a24defbdec5e67e891234dacd345d6",
    };
    const scales = [rows * k / block_size]f32{
        384, 208, 384, 144, 88,  256, 240, 176,
        256, 256, 0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,
        352, 0,   0,   384, 208, 256, 256, 240,
        13,  256, 8,   22,  256, 104, 13,  26,
        256, 32,  6,   96,  256, 176, 256, 22,
        13,  96,  26,  104, 256, 112, 6.5, 128,
        22,  64,  320, 256, 72,  320, 36,  13,
    };
    const amax = [rows]f32{ 1.9453125, 1016, 0, 936, 5.5, 48, 320, 3584 };

    const Local = struct {
        const Outputs = struct { values: Tensor, scales: Tensor, global_scale: Tensor };

        fn forward(x: Tensor) Outputs {
            // Along the first axis: nothing may rely on the quantized axis being the last.
            const input = quantize(x.transpose(.{ .k, .row }), .k, .{ .global_scale = .per_row, .block_scale = .best_of_four_or_six });
            return .{
                // As bits: the sign of a zero is part of its code.
                .values = input.values.convert(.f32).bitCast(.u32).transpose(.{ .row, .k }),
                .scales = input.scales.convert(.f32).transpose(.{ .row, .sc }),
                .global_scale = input.global_scale.?,
            };
        }

        fn hash(i: usize) u32 {
            var h = @as(u32, @intCast(i)) *% 2654435761 +% 12345;
            h ^= h >> 15;
            h *%= 2246822519;
            return h ^ (h >> 13);
        }
    };
    // Rows 0 and 3 are noise over 8 binades; row 3 has a zero block and a block too small for
    // E4M3 next to its row. Row 1 holds, next to its row's maximum, a block whose amax/4 and amax/6
    // errors tie when summed over adjacent pairs and are one ulp apart in the order of a CUDA
    // reduction, which picks amax/4. Row 2 is zero. In rows 4 to 7 each value is its block's
    // maximum times n/16, which lands the scaled values on E2M1 rounding ties.
    const order_sensitive = [block_size]u16{ 0xc1f7, 0xc1f1, 0x41b1, 0x408e, 0x4229, 0xc375, 0xc286, 0x4473, 0x413d, 0x41b5, 0x413e, 0xc3cb, 0xc472, 0x40ce, 0xc477, 0xc390 };
    const fractions = [block_size]f32{ 16, 1, 2, 3, 5, 7, 10, 14, 4, 8, 12, 6, 9, 11, 13, 15 };
    var host: [rows * k]u32 = undefined;
    for (&host, 0..) |*fp32, i| {
        const row = i / k;
        const block = i % k / block_size;
        const h = Local.hash(i);
        // The BF16 value vLLM quantized.
        const bf16: u32 = if (row == 1)
            (if (block == 0) order_sensitive[i % block_size] else if (i % k == block_size) 0x447e else 0)
        else word: {
            const magnitude: f32 = if (row == 2 or (row == 3 and block == 1))
                0
            else if (row < 4)
                std.math.ldexp(@as(f32, @floatFromInt(128 + ((h >> 20) & 0x7f))), @as(i32, @intCast(3 * row + ((h >> 28) & 7))) - @as(i32, if (row == 3 and block == 2) 34 else 14))
            else ties: {
                const hb = Local.hash(1000 + row * 8 + block);
                break :ties std.math.ldexp(@as(f32, @floatFromInt(8 + (hb & 7))) * fractions[i % block_size], @as(i32, @intCast((hb >> 8) % 6 + 3 * (row - 4))) - 10);
            };
            // Every value has at most 8 significant bits: BF16 holds it exactly.
            break :word @as(u32, @bitCast(if (h >> 31 != 0 and magnitude != 0) -magnitude else magnitude)) >> 16;
        };
        // The quantizer reads FP32 as BF16: the nearest FP32 values on either side of the BF16
        // rounding boundaries quantize like the BF16 value between them.
        fp32.* = if (bf16 & 0x7fff == 0) bf16 << 16 else if (i % 2 == 0) (bf16 << 16) | 0x7fff else (bf16 << 16) - 0x7fff;
    }

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const x: Tensor = .init(.{ .row = rows, .k = k }, .f32);
    var exe = try platform.compileFn(allocator, io, Local.forward, .{x}, .{});
    defer exe.deinit();
    var buffer = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&host));
    defer buffer.deinit();
    var output: zml.Bufferized(Local.Outputs) = undefined;
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    runner.run(io, .{buffer}, .{&output}, .{ .wait = true });
    defer zml.Buffer.deinitAll(Local.Outputs, &output);
    var actual_values = try output.values.toSliceAlloc(allocator, io);
    defer actual_values.free(allocator);
    var actual_scales = try output.scales.toSliceAlloc(allocator, io);
    defer actual_scales.free(allocator);
    var actual_global = try output.global_scale.toSliceAlloc(allocator, io);
    defer actual_global.free(allocator);

    const e2m1 = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    var values: [rows * k]u32 = undefined;
    for (&values, 0..) |*value, i| {
        const code = try std.fmt.charToDigit(codes[i / k][i % k], 16);
        value.* = @bitCast(if (code & 8 != 0) -e2m1[code & 7] else e2m1[code & 7]);
    }
    var global: [rows]f32 = undefined;
    for (&global, amax) |*scale, maximum| scale.* = maximum * @as(f32, 1.0 / 1536.0);
    try std.testing.expectEqualSlices(f32, &global, actual_global.constItems(f32));
    try std.testing.expectEqualSlices(f32, &scales, actual_scales.constItems(f32));
    try std.testing.expectEqualSlices(u32, &values, actual_values.constItems(u32));
}
