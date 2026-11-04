//! Quantization representations, metadata, and tensor conversion.
const std = @import("std");

const stdx = @import("stdx");

const zml = @import("zml.zig");

const DataType = @import("dtype.zig").DataType;
const platform_mod = @import("platform.zig");
const Platform = platform_mod.Platform;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

pub const nvfp4_block_size = 16;
pub const mx_block_size = 32;

pub const Quantization = struct {
    scheme: Quantization.Scheme, // decided when the checkpoint is read
    scales: Tensor,
    global_scale: ?Quantization.GlobalScale = null,
    input_scale: ?Quantization.GlobalScale = null,
    /// HACK: `scales` is stored in the 128x4 blocked layout of Blackwell's block-scaled mma.
    // Something cleaner will need to be done to differentiate different kinds of swizzling
    swizzled_scales: bool = false,

    /// A per-tensor scale and the polarity its producer wrote it in
    pub const GlobalScale = struct {
        value: Tensor,
        operation: Operation,

        pub const Operation = enum { multiply, divide };

        pub fn asMultiplier(self: Quantization.GlobalScale) Tensor {
            const s = self.value.convert(.f32);
            const flat = if (s.shape().count() == 1) s.asScalar() else s; // essentially normalization

            return switch (self.operation) {
                .multiply => flat,
                .divide => Tensor.scalar(1.0, .f32).broad(flat.shape()).div(flat),
            };
        }
    };

    pub const Scheme = enum {
        /// f4e2m1 values (`u8`-packed or native), f8e4m3fn scale per 16 contracted values.
        /// Emitted by llm-compressor and by NVIDIA's ModelOpt.
        nvfp4,
        /// f8e4m3fn values, one e8m0 (power-of-two) scale per 32 contracted values
        mxfp8,
        /// f4e2m1 values (`u8`-packed or native), e8m0 scale per 32 contracted values.
        // GPT-OSS like
        mxfp4,
        /// f8e4m3fn values, one bf16 or f32 scale per output channel, constant along the contraction.
        /// Emitted by llm-compressor, including for the layers an NVFP4 recipe leaves in FP8.
        fp8_per_channel,
        /// f8e4m3fn, f8e4m3fnuz, or f8e8m0 values, one bf16 or f32 scale per 128x128 tile.
        /// The DeepSeek-style FP8 that model vendors publish themselves, under `weight_scale_inv`.
        fp8_block128,
        /// E4M3FN or E4M3FNUZ weights with E8M0, BF16 or F32 scales per 32x32 tile.
        fp8_block32,
        /// f8e4m3fn values, one scale for the whole tensor. Spelled `[1, 1]` rather than as a
        /// scalar: XLA's composite rewriter requires the scale to have the operand's rank.
        fp8_per_tensor,

        pub fn accepts(self: Quantization.Scheme, weight: Shape, scale: Shape) bool {
            if (weight.rank() != 2) return false;

            const n = weight.dim(0);
            const k = if (isPackedFp4(self, weight.dtype())) 2 * weight.dim(1) else weight.dim(1);

            return switch (self) {
                .nvfp4 => (weight.dtype() == .u8 or weight.dtype() == .f4e2m1) and
                    scale.dtype() == .f8e4m3fn and scale.rank() == 2 and
                    scale.dim(0) == n and scale.dim(1) * nvfp4_block_size == k,
                .mxfp8 => weight.dtype() == .f8e4m3fn and isMxScale(scale) and
                    scale.dim(0) == n and scale.dim(1) * mx_block_size == k,
                .mxfp4 => (weight.dtype() == .u8 or weight.dtype() == .i8 or weight.dtype() == .f4e2m1) and
                    isMxScale(scale) and
                    scale.dim(0) == n and scale.dim(1) * mx_block_size == k,
                .fp8_per_tensor => weight.dtype() == .f8e4m3fn and scale.count() == 1,
                .fp8_per_channel => weight.dtype() == .f8e4m3fn and
                    (scale.dtype() == .bf16 or scale.dtype() == .f32) and
                    scale.count() > 1 and scale.rank() == 2 and
                    scale.dim(0) == n and scale.dim(1) == 1,
                .fp8_block32 => (weight.dtype() == .f8e4m3fn or weight.dtype() == .f8e4m3fnuz) and
                    (scale.dtype() == .f8e8m0 or ((scale.dtype() == .bf16 or scale.dtype() == .f32) and scale.count() > 1)) and
                    scale.rank() == 2 and scale.dim(0) * 32 == n and scale.dim(1) * 32 == k,
                .fp8_block128 => (weight.dtype() == .f8e4m3fn or weight.dtype() == .f8e4m3fnuz or weight.dtype() == .f8e8m0) and
                    (scale.dtype() == .bf16 or scale.dtype() == .f32) and
                    scale.count() > 1 and scale.rank() == 2 and
                    scale.dim(0) == (std.math.divCeil(i64, n, 128) catch unreachable) and
                    scale.dim(1) == (std.math.divCeil(i64, k, 128) catch unreachable),
            };
        }

        pub fn classify(weight: Shape, scale: Shape) ?Quantization.Scheme {
            for (std.enums.values(Quantization.Scheme)) |scheme| {
                if (scheme.accepts(weight, scale)) return scheme;
            }

            return null;
        }

        pub fn isMx(self: Quantization.Scheme) bool {
            return self == .mxfp8 or self == .mxfp4;
        }
    };
};

/// Logical quantized values and the scales needed to reconstruct them.
pub const QuantizedTensor = struct {
    values: Tensor,
    scales: Scales,

    pub const ValidationError = error{
        InvalidValueDtype,
        InvalidScaleDtype,
        InvalidGroupSize,
        InvalidScaleShape,
        UnsupportedScaleLayout,
    };

    /// Validate logical values and a plain scale grid in the values' axis order.
    /// A single scale may broadcast over all values. Otherwise, axis scaling
    /// reduces at most one dimension and tile scaling at most two.
    pub fn validate(self: QuantizedTensor) ValidationError!void {
        const values = self.values.shape();
        const scales = self.scales;
        if (scales.swizzled) return error.UnsupportedScaleLayout;
        if (!values.dtype().isFloat()) return error.InvalidValueDtype;
        if (!scales.values.dtype().isFloat()) return error.InvalidScaleDtype;
        switch (scales.scaling) {
            .axis => |size| if (size == 0) return error.InvalidGroupSize,
            .tile => |size| if (size == 0) return error.InvalidGroupSize,
            .tensor => {},
        }
        if (scales.values.count() == 1) return;
        if (scales.values.rank() != values.rank()) return error.InvalidScaleShape;
        var reduced_axes: usize = 0;
        var grouped_axes: usize = 0;
        for (0..values.rank()) |axis| {
            const block: i64 = switch (scales.scaling) {
                .tensor => values.dim(axis),
                .axis => |size| size orelse values.dim(axis),
                .tile => |size| size,
            };
            if (block <= 0) return error.InvalidGroupSize;
            const groups = std.math.divCeil(i64, values.dim(axis), block) catch return error.InvalidScaleShape;
            if (scales.values.dim(axis) == groups) grouped_axes += 1;
            if (scales.values.dim(axis) == values.dim(axis)) continue;
            if (scales.values.dim(axis) != groups) return error.InvalidScaleShape;
            reduced_axes += 1;
        }
        switch (scales.scaling) {
            .axis => if (reduced_axes > 1 or grouped_axes < 1) return error.InvalidScaleShape,
            .tile => if (reduced_axes > 2 or grouped_axes < 2) return error.InvalidScaleShape,
            .tensor => if (grouped_axes < 1) return error.InvalidScaleShape,
        }
    }

    pub fn reuseBuffer(self: QuantizedTensor, other: QuantizedTensor) QuantizedTensor {
        return .{
            .values = self.values.reuseBuffer(other.values),
            .scales = self.scales.reuseBuffer(other.scales),
        };
    }
};

pub const Scales = struct {
    /// Logical floating-point scale values. Plain grids follow the operand's axis order.
    values: Tensor,
    scaling: Scaling,
    global: ?GlobalScale = null,
    /// Scales use Blackwell's 128x4 blocked MMA layout.
    swizzled: bool = false,

    pub fn reuseBuffer(self: Scales, other: Scales) Scales {
        var result = self;
        result.values = self.values.reuseBuffer(other.values);
        if (result.global) |*scale| {
            if (other.global) |previous| scale.value = scale.value.reuseBuffer(previous.value);
        }
        return result;
    }
};

/// An outer scale using the existing producer multiplier/divisor convention.
pub const GlobalScale = Quantization.GlobalScale;

pub const Scaling = union(enum) {
    /// One scale over the selected axes. Unselected axes remain independent.
    tensor,
    /// Positive group size along one selected axis; null covers that entire axis.
    axis: ?u32,
    /// Positive side length of a square tile over two selected axes.
    tile: u32,

    pub fn eql(self: Scaling, other: Scaling) bool {
        return std.meta.eql(self, other);
    }
};

/// Logical value encoding and scale coverage.
/// Consumers bind scaling to tensor axes and handle physical storage layouts.
pub const Scheme = struct {
    values: DataType,
    scales: DataType,
    scaling: Scaling,

    pub const nvfp4: Scheme = .{ .values = .f4e2m1, .scales = .f8e4m3fn, .scaling = .{ .axis = nvfp4_block_size } };
    pub const mxfp4: Scheme = .{ .values = .f4e2m1, .scales = .f8e8m0, .scaling = .{ .axis = mx_block_size } };
    pub const mxfp8: Scheme = .{ .values = .f8e4m3fn, .scales = .f8e8m0, .scaling = .{ .axis = mx_block_size } };

    pub fn eql(a: ?Scheme, b: ?Scheme) bool {
        if (a == null or b == null) return a == null and b == null;
        return std.meta.eql(a.?, b.?);
    }

    pub fn isTiled(self: ?Scheme, size: u32) bool {
        return switch ((self orelse return false).scaling) {
            .tile => |tile| tile == size,
            else => false,
        };
    }

    pub fn isPerAxis(self: ?Scheme) bool {
        return switch ((self orelse return false).scaling) {
            .axis => |size| size == null,
            else => false,
        };
    }

    pub fn isPerTensor(self: ?Scheme) bool {
        return (self orelse return false).scaling == .tensor;
    }
};

/// Reconstruct logical values in FP32, then convert to dtype.
/// Scales use the same axis order as values, with one entry per group or tile;
/// a scalar scale broadcasts over the entire tensor. Unpack values and unswizzle
/// scales first. Values and scales must carry their logical floating-point dtypes.
pub fn dequantize(input: QuantizedTensor, dtype: DataType) Tensor {
    const values = input.values;
    const scales = input.scales;
    input.validate() catch |err| stdx.debug.panic("dequantize: {s} for scaling {any}, values {f}, scales {f}", .{ @errorName(err), scales.scaling, values, scales.values });
    var expanded = scales.values.convert(.f32);
    if (expanded.count() == 1) {
        expanded = expanded.asScalar();
    } else {
        for (0..values.rank()) |axis| {
            if (expanded.dim(axis) == values.dim(axis)) continue;
            const block: i64 = switch (scales.scaling) {
                .tensor => values.dim(axis),
                .axis => |size| size orelse values.dim(axis),
                .tile => |size| size,
            };
            const groups = std.math.divCeil(i64, values.dim(axis), block) catch unreachable;
            const shape = expanded.shape().insert(axis + 1, .{block});
            expanded = expanded.broadcast(shape, Shape.range(shape.rank(), .i64).remove(axis + 1).dims())
                .reshape(expanded.shape().setDim(axis, groups * block))
                .slice(axis, .{ .end = values.dim(axis) });
        }
    }
    var result = values.convert(.f32).mul(expanded.broad(values.shape().withDtype(.f32)));
    if (scales.global) |scale| result = result.mul(scale.asMultiplier().broad(result.shape()));
    return result.convert(dtype);
}

pub const QuantizedInput = struct {
    values: Tensor,
    scales: Tensor,
    global_scale: ?Tensor,

    pub fn reuseBuffer(self: QuantizedInput, other: QuantizedInput) QuantizedInput {
        return .{
            .values = self.values.reuseBuffer(other.values),
            .scales = self.scales.reuseBuffer(other.scales),
            .global_scale = if (self.global_scale != null and other.global_scale != null) self.global_scale.?.reuseBuffer(other.global_scale.?) else self.global_scale,
        };
    }
};

pub fn quantizeInput(quantization: Quantization, input: Tensor, axis: Shape.Tag, platform: *const Platform) ?QuantizedInput {
    const global_scale: ?Tensor = if (quantization.input_scale) |scale| scale.asMultiplier() else null;

    return switch (quantization.scheme) {
        .nvfp4 => if (supportsNvfp4InputQuantization(platform)) quantizeNvfp4(input.convert(.bf16), global_scale, axis) else null,
        .fp8_block32, .fp8_block128 => blk: {
            const dtype: DataType = switch (platform.target) {
                .cuda => .f8e4m3fn,
                .rocm => if (platform_mod.rocm.computeCapability(platform)) |capability| switch (capability.architecture()) {
                    .cdna4, .rdna4 => .f8e4m3fn,
                    .cdna3 => .f8e4m3fnuz,
                    .cdna1, .cdna2, .rdna2, .rdna3, .rdna3_5 => return null,
                } else return null,
                else => return null,
            };

            break :blk switch (quantization.scheme) {
                .fp8_block32 => quantizeBlockFp8(input, axis, 32, dtype, .f8e8m0),
                .fp8_block128 => quantizeBlockFp8(input.convert(.bf16), axis, 128, dtype, .f32),
                else => unreachable,
            };
        },
        .mxfp8, .mxfp4, .fp8_per_channel, .fp8_per_tensor => null,
    };
}

pub fn quantizeMxfp4(x: Tensor, axis: anytype) QuantizedInput {
    stdx.debug.assert(@mod(x.dim(axis), mx_block_size) == 0, "MXFP4 activation width must be divisible by {}, got {f}", .{ mx_block_size, x.shape() });

    const grouped = x.convert(.f32).splitAxis(axis, .{ .mx_ks = -1, .mx_block = mx_block_size });
    const raw_scale = grouped.abs().max(.mx_block).maximum(.scalar(6 * 0x1p-126, .f32)).scale(1.0 / 6.0);
    const scales = ceilPowerOfTwo(raw_scale);

    return .{
        .values = grouped.div(scales.broad(grouped.shape()))
            .clamp(.scalar(-6, .f32), .scalar(6, .f32))
            .convert(.f4e2m1)
            .reshape(x.shape().withDtype(.f4e2m1)),
        .scales = scales.reshape(x.shape().setDim(axis, @divExact(x.dim(axis), mx_block_size)).withDtype(.f32)).convert(.f8e8m0),
        .global_scale = null,
    };
}

pub fn quantizeMxfp8(x: Tensor, axis: anytype) QuantizedInput {
    const platform = zml.Compiler.current().platform;

    const dtype: ?DataType = switch (platform.target) {
        .cuda, .cpu, .tpu => .f8e4m3fn,
        .rocm => if (platform_mod.rocm.computeCapability(platform)) |capability| switch (capability.architecture()) {
            .cdna4, .rdna4 => .f8e4m3fn,
            .cdna3 => .f8e4m3fnuz,
            .cdna1, .cdna2, .rdna2, .rdna3, .rdna3_5 => null,
        } else null,
        else => null,
    };

    if (dtype == null) std.debug.panic("MXFP8 is not supported on {}", .{platform.target});
    return quantizeBlockFp8(x, axis, mx_block_size, dtype.?, .f8e8m0);
}

fn ceilPowerOfTwo(raw_scale: Tensor) Tensor {
    const bits = raw_scale.bitCast(.u32);
    return bits.addConstant(0x7fffff).logical(.AND, .scalar(0xff800000, .u32)).bitCast(.f32);
}

/// Quantize activation blocks, preserving the input's axes.
/// E8M0 scales round upward to a power of two to avoid overflowing FP8 values.
pub fn quantizeBlockFp8(x: Tensor, axis: anytype, block_size: i64, dtype: DataType, scale_dtype: DataType) QuantizedInput {
    stdx.debug.assert(block_size > 0, "block FP8 size must be positive, got {}", .{block_size});
    stdx.debug.assert(@mod(x.dim(axis), block_size) == 0, "block FP8 activation width must be divisible by {}, got {f}", .{ block_size, x.shape() });
    // Match the routed-MoE quantizer's FNUZ range.
    const fp8_max: f32 = switch (dtype) {
        .f8e4m3fn => 448.0,
        .f8e4m3fnuz => 224.0,
        else => stdx.debug.panic("expected E4M3 FP8 dtype, got {s}", .{@tagName(dtype)}),
    };
    const grouped = x.convert(.f32).splitAxis(axis, .{ .fp8_ks = -1, .fp8_block = block_size });
    const scales = switch (scale_dtype) {
        // Put scale arithmetic before the reduction so it can fuse into the producer.
        .f32 => grouped.abs()
            .maximum(.scalar(1e-10, .f32))
            .scale(1.0 / fp8_max)
            .max(.fp8_block),
        .f8e8m0 => blk: {
            const raw_scale = grouped.abs().maximum(.scalar(1e-4, .f32)).scale(1.0 / fp8_max).max(.fp8_block);
            break :blk ceilPowerOfTwo(raw_scale);
        },
        else => stdx.debug.panic("expected F32 or E8M0 scale dtype, got {s}", .{@tagName(scale_dtype)}),
    };
    const values = grouped.div(scales.broad(grouped.shape()))
        .clamp(.scalar(-fp8_max, .f32), .scalar(fp8_max, .f32))
        .convert(dtype)
        .reshape(x.shape().withDtype(dtype));
    return .{
        .values = values,
        .scales = scales.reshape(x.shape().setDim(axis, @divExact(x.dim(axis), block_size)).withDtype(.f32)).convert(scale_dtype),
        .global_scale = null,
    };
}

pub fn quantizeNvfp4(x: Tensor, input_global_scale: ?Tensor, axis: anytype) QuantizedInput {
    const value_max = 6.0;
    const scale_min_normal = 0x1p-6;
    const scale_max = DataType.f8e4m3fn.maxValue().as(f32);

    stdx.debug.assert(x.shape().hasTag(axis) != null, "quantizeNvfp4 expects x to have {any} tag, got {f}", .{ axis, x.shape() });
    stdx.debug.assert(@rem(x.dim(axis), nvfp4_block_size) == 0, "quantizeNvfp4 expects {any} to be a multiple of {}, got {f}", .{ axis, nvfp4_block_size, x.shape() });

    const dt = x.dtype();
    const scaled = if (input_global_scale) |igs|
        x.div(igs.convert(dt).broad(x.shape()))
    else
        x;
    const grouped = scaled.splitAxis(axis, .{ .sc = -1, .blk = nvfp4_block_size });

    const amax = grouped.abs()
        .scale(1.0 / value_max)
        .clamp(.scalar(scale_min_normal, dt), .scalar(scale_max, dt))
        .max(.blk);

    const scales = amax.convert(.f8e4m3fn);

    const divisor = scales.convert(dt)
        .maximum(.scalar(scale_min_normal, dt))
        .broad(grouped.shape());

    const values = grouped.div(divisor)
        .convert(.f4e2m1)
        .reshape(x.shape().withDtype(.f4e2m1));

    return .{
        .values = values,
        .scales = scales.squeeze(.blk),
        .global_scale = input_global_scale,
    };
}

fn supportsNvfp4InputQuantization(platform: *const Platform) bool {
    return if (platform_mod.cuda.computeCapability(platform)) |cc| cc.atLeast(.{ .major = 10, .minor = 0 }) else false;
}

fn isPackedFp4(scheme: ?Quantization.Scheme, weight_dtype: DataType) bool {
    return (scheme == .nvfp4 or scheme == .mxfp4) and (weight_dtype == .u8 or weight_dtype == .i8);
}

fn isMxScale(scale: Shape) bool {
    return scale.rank() == 2 and (scale.dtype() == .f8e8m0 or scale.dtype() == .u8);
}

// Note: this test will evolve as we support more (INT4/8 and FP8 block-128 as a
// scaled-dot are still open; see the scaledDot doc comment).
test "Quantization.Scheme.classify" {
    const expect = std.testing.expectEqual;

    // unsloth/Qwen3.6-27B-NVFP4: compressed-tensors, carrying both of its schemes under
    // `weight_scale` -- NVFP4 on the MLPs of layers 0-55, FP8 per-channel everywhere else.
    const nvfp4_packed: Shape = .init(.{ .dout = 17408, .d = 2560 }, .u8);
    try expect(@as(?Quantization.Scheme, .nvfp4), Quantization.Scheme.classify(nvfp4_packed, .init(.{ .dout = 17408, .sc = 320 }, .f8e4m3fn)));
    try expect(@as(?Quantization.Scheme, .fp8_per_channel), Quantization.Scheme.classify(
        .init(.{ .dout = 17408, .d = 5120 }, .f8e4m3fn),
        .init(.{ .dout = 17408, .sc = 1 }, .bf16),
    ));

    // nvidia/Gemma-4-31B-IT-NVFP4: ModelOpt, packed values under the plain `weight` name.
    try expect(@as(?Quantization.Scheme, .nvfp4), Quantization.Scheme.classify(
        .init(.{ .dout = 21504, .d = 2688 }, .u8),
        .init(.{ .dout = 21504, .sc = 336 }, .f8e4m3fn),
    ));

    // Native (unpacked) f4e2m1, with K no longer doubled.
    try expect(@as(?Quantization.Scheme, .nvfp4), Quantization.Scheme.classify(
        .init(.{ .dout = 17408, .d = 5120 }, .f4e2m1),
        .init(.{ .dout = 17408, .sc = 320 }, .f8e4m3fn),
    ));

    // Qwen/Qwen3.6-27B-FP8: block 128, spelled `weight_scale_inv`.
    try expect(@as(?Quantization.Scheme, .fp8_block128), Quantization.Scheme.classify(
        .init(.{ .dout = 17408, .d = 5120 }, .f8e4m3fn),
        .init(.{ .dout = 136, .sc = 40 }, .bf16),
    ));
    try expect(@as(?Quantization.Scheme, .fp8_block128), Quantization.Scheme.classify(
        .init(.{ .dout = 5120, .d = 6144 }, .f8e4m3fn),
        .init(.{ .dout = 40, .sc = 48 }, .bf16),
    ));

    inline for (.{ .f8e4m3fn, .f8e4m3fnuz }) |dtype| {
        try expect(@as(?Quantization.Scheme, .fp8_block32), Quantization.Scheme.classify(
            .init(.{ .dout = 1280, .d = 5120 }, dtype),
            .init(.{ .dout = 40, .sc = 160 }, .f8e8m0),
        ));
    }

    // GLM uses f32 scales and permits a partial final 128-row tile.
    try expect(@as(?Quantization.Scheme, .fp8_block128), Quantization.Scheme.classify(
        .init(.{ .dout = 2048, .d = 6144 }, .f8e4m3fn),
        .init(.{ .dout = 16, .sc = 48 }, .f32),
    ));
    try expect(@as(?Quantization.Scheme, .fp8_block128), Quantization.Scheme.classify(
        .init(.{ .dout = 576, .d = 6144 }, .f8e4m3fn),
        .init(.{ .dout = 5, .sc = 48 }, .f32),
    ));
    try expect(@as(?Quantization.Scheme, .fp8_block128), Quantization.Scheme.classify(
        .init(.{ .dout = 576, .d = 6144 }, .f8e8m0),
        .init(.{ .dout = 5, .sc = 48 }, .f32),
    ));

    // Mistral's per-tensor FP8: one scale for the whole tensor, rank 0 or [1].
    try expect(@as(?Quantization.Scheme, .fp8_per_tensor), Quantization.Scheme.classify(.init(.{ .dout = 4096, .d = 4096 }, .f8e4m3fn), .init(.{}, .f32)));
    try expect(@as(?Quantization.Scheme, .fp8_per_tensor), Quantization.Scheme.classify(.init(.{ .dout = 4096, .d = 4096 }, .f8e4m3fn), .init(.{ .g = 1 }, .f32)));

    // RadixArk/Muse-Glimmer-NVFP4: ModelOpt's MIXED_PRECISION recipe, whose MXFP8 half
    // lands on down_proj and lm_head. safetensors has no e8m0 dtype, so the scale arrives
    // as u8 under `weight_scale_inv` -- the shape below is verbatim from the checkpoint.
    try expect(@as(?Quantization.Scheme, .mxfp8), Quantization.Scheme.classify(
        .init(.{ .dout = 6656, .d = 19968 }, .f8e4m3fn),
        .init(.{ .dout = 6656, .sc = 624 }, .u8),
    ));
    try expect(@as(?Quantization.Scheme, .mxfp8), Quantization.Scheme.classify(
        .init(.{ .dout = 202048, .d = 6656 }, .f8e4m3fn),
        .init(.{ .dout = 202048, .sc = 208 }, .u8),
    ));
    // The same tensor from a store that types the scale natively.
    try expect(@as(?Quantization.Scheme, .mxfp8), Quantization.Scheme.classify(
        .init(.{ .dout = 6656, .d = 19968 }, .f8e4m3fn),
        .init(.{ .dout = 6656, .sc = 624 }, .f8e8m0),
    ));
    // MXFP8 rejects any group but 32: 19968 / 16 = 1248, / 64 = 312.
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(
        .init(.{ .dout = 6656, .d = 19968 }, .f8e4m3fn),
        .init(.{ .dout = 6656, .sc = 1248 }, .u8),
    ));
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(
        .init(.{ .dout = 6656, .d = 19968 }, .f8e4m3fn),
        .init(.{ .dout = 6656, .sc = 312 }, .u8),
    ));

    // MXFP4: the same u8 packing as NVFP4 (two nibbles per byte, K halved on disk), with
    // an e8m0 scale per 32 -- so K = 2 * 2560 = 5120 and 5120 / 32 = 160.
    try expect(@as(?Quantization.Scheme, .mxfp4), Quantization.Scheme.classify(nvfp4_packed, .init(.{ .dout = 17408, .sc = 160 }, .u8)));
    try expect(@as(?Quantization.Scheme, .mxfp4), Quantization.Scheme.classify(nvfp4_packed, .init(.{ .dout = 17408, .sc = 160 }, .f8e8m0)));
    // DeepSeek V4 stores the same packed FP4 bits in signed bytes.
    try expect(@as(?Quantization.Scheme, .mxfp4), Quantization.Scheme.classify(
        .init(.{ .dout = 2048, .d = 2048 }, .i8),
        .init(.{ .dout = 2048, .sc = 128 }, .f8e8m0),
    ));
    // Native (unpacked) f4e2m1, K no longer halved.
    try expect(@as(?Quantization.Scheme, .mxfp4), Quantization.Scheme.classify(
        .init(.{ .dout = 17408, .d = 5120 }, .f4e2m1),
        .init(.{ .dout = 17408, .sc = 160 }, .f8e8m0),
    ));
    // An e8m0 scale never turns a non-fp4/fp8 weight into MX.
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(
        .init(.{ .dout = 17408, .d = 5120 }, .bf16),
        .init(.{ .dout = 17408, .sc = 160 }, .f8e8m0),
    ));

    // Rejected: everything no backend here can express.
    const fp8: Shape = .init(.{ .dout = 10240, .d = 5120 }, .f8e4m3fn);
    // Group 32 with an e4m3 scale: neither NVFP4 (wrong group) nor MX (wrong scale type).
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(nvfp4_packed, .init(.{ .dout = 17408, .sc = 160 }, .f8e4m3fn)));
    // Group 16 with an e8m0 scale: the mirror image, and equally unclaimed.
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(nvfp4_packed, .init(.{ .dout = 17408, .sc = 320 }, .f8e8m0)));
    // Per-channel in f32: what fusing per-tensor projections produces. The Metal
    // per-channel kernels have f32 entries, so this is a real scheme, not a decline.
    try expect(@as(?Quantization.Scheme, .fp8_per_channel), Quantization.Scheme.classify(fp8, .init(.{ .dout = 10240, .sc = 1 }, .f32)));
    // A block grid that is neither per-channel nor 128x128.
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(fp8, .init(.{ .dout = 160, .sc = 80 }, .bf16)));
    // Stacked MoE weights are rank 3 and go through the MoE path, not this one.
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(
        .init(.{ .expert = 128, .dout = 1408, .d = 1408 }, .u8),
        .init(.{ .expert = 128, .dout = 1408, .sc = 176 }, .f8e4m3fn),
    ));

    // `fp8_per_tensor` never checks the scale dtype, so a [1, 1] u8 scale on a [1, 32]
    // weight satisfies it as well as `mxfp8`. Declaration order is what decides, and
    // `mxfp8` is declared first
    try expect(@as(?Quantization.Scheme, .mxfp8), Quantization.Scheme.classify(
        .init(.{ .dout = 1, .d = 32 }, .f8e4m3fn),
        .init(.{ .dout = 1, .sc = 1 }, .u8),
    ));

    // A one-tile block grid is indistinguishable from a per-tensor scale by shape alone; the
    // `count() > 1` guard on the grids is what makes per-tensor win.
    try expect(@as(?Quantization.Scheme, .fp8_per_tensor), Quantization.Scheme.classify(
        .init(.{ .dout = 128, .d = 128 }, .f8e4m3fn),
        .init(.{ .dout = 1, .sc = 1 }, .bf16),
    ));
}

test "block FP8 quantization preserves axes and reconstructs constant blocks in FN and FNUZ" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    inline for (.{ DataType.f8e4m3fn, DataType.f8e4m3fnuz }) |dtype| {
        const Local = struct {
            const Outputs = struct { values: Tensor, scales: Tensor };
            fn forward(x: Tensor) Outputs {
                const input = quantizeBlockFp8(x, .k, 128, dtype, .f32);
                return .{ .values = input.values.convert(.f32), .scales = input.scales };
            }
        };
        const x: Tensor = .init(.{ .b = 2, .k = 256, .s = 3 }, .bf16);
        var exe = try platform.compileFn(allocator, io, Local.forward, .{x}, .{});
        defer exe.deinit();
        try zml.testing.expectEqualShapes(x.shape().withDtype(.f32), exe.output_shapes[0]);
        try zml.testing.expectEqualShapes(x.shape().setDim(.k, 2).withDtype(.f32), exe.output_shapes[1]);
        var host: [2 * 256 * 3]zml.floats.BFloat16 = undefined;
        for (&host, 0..) |*value, i| {
            const magnitude: f32 = if ((i / 3) % 256 < 128) 1.0 else 2.0;
            value.* = .fromF32(switch (i % 3) {
                0 => 0.0,
                1 => magnitude,
                else => -magnitude,
            });
        }
        var buffer = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&host));
        defer buffer.deinit();
        var output: zml.Bufferized(Local.Outputs) = undefined;
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        runner.run(io, .{buffer}, .{&output}, .{ .wait = true });
        defer zml.Buffer.deinitAll(Local.Outputs, &output);
        var values = try output.values.toSliceAlloc(allocator, io);
        defer values.free(allocator);
        var scales = try output.scales.toSliceAlloc(allocator, io);
        defer scales.free(allocator);
        for (scales.constItems(f32)) |scale| try std.testing.expect(std.math.isFinite(scale) and scale > 0);
        for (values.constItems(f32), 0..) |value, i| {
            const scale_index = (i / (256 * 3)) * 6 + ((i / 3) % 256) / 128 * 3 + i % 3;
            try std.testing.expectApproxEqAbs(host[i].toF32(), value * scales.constItems(f32)[scale_index], 1e-6);
        }
    }
}

test "block FP8 E8M0 scales preserve power-of-two boundaries and nonminor axes" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    inline for (.{ DataType.f8e4m3fn, DataType.f8e4m3fnuz }) |dtype| {
        const Local = struct {
            const Outputs = struct { values: Tensor, scales: Tensor };
            fn forward(x: Tensor) Outputs {
                const input = quantizeBlockFp8(x, .k, 32, dtype, .f8e8m0);
                return .{ .values = input.values.convert(.f32), .scales = input.scales.convert(.f32) };
            }
        };
        const x: Tensor = .init(.{ .b = 2, .k = 256, .s = 3 }, .f32);
        var exe = try platform.compileFn(allocator, io, Local.forward, .{x}, .{});
        defer exe.deinit();
        try zml.testing.expectEqualShapes(x.shape().withDtype(.f32), exe.output_shapes[0]);
        try zml.testing.expectEqualShapes(x.shape().setDim(.k, 8).withDtype(.f32), exe.output_shapes[1]);
        const fp8_max: f32 = if (dtype == .f8e4m3fn) 448.0 else 224.0;
        var host: [2 * 256 * 3]f32 = undefined;
        var maxima: [2 * 8 * 3]f32 = @splat(0);
        for (&host, 0..) |*value, i| {
            const block = (i / 3) % 256 / 32;
            const scale_index = (i / (256 * 3)) * 24 + block * 3 + i % 3;
            const boundary = fp8_max * std.math.pow(f32, 2, @as(f32, @floatFromInt(block)) - 4);
            const bits: u32 = @bitCast(boundary);
            const magnitude: f32 = switch (i % 3) {
                0 => @bitCast(bits - 1),
                1 => boundary,
                else => @bitCast(bits + 1),
            };
            value.* = if (block == 0) 0 else if (block == 1) 1e-30 else magnitude * (if (i % 2 == 0) @as(f32, 1) else -1);
            maxima[scale_index] = @max(maxima[scale_index], @abs(value.*));
        }
        var expected_scales: [2 * 8 * 3]f32 = undefined;
        for (&expected_scales, maxima) |*scale, maximum| {
            const raw = @max(maximum, 1e-4) * (1.0 / fp8_max);
            const bits: u32 = @bitCast(raw);
            const exponent = (bits >> 23) + @intFromBool(bits & 0x7fffff != 0);
            scale.* = @bitCast(exponent << 23);
        }
        var buffer = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&host));
        defer buffer.deinit();
        var output: zml.Bufferized(Local.Outputs) = undefined;
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        runner.run(io, .{buffer}, .{&output}, .{ .wait = true });
        defer zml.Buffer.deinitAll(Local.Outputs, &output);
        var values = try output.values.toSliceAlloc(allocator, io);
        defer values.free(allocator);
        var scales = try output.scales.toSliceAlloc(allocator, io);
        defer scales.free(allocator);
        try std.testing.expectEqualSlices(f32, &expected_scales, scales.constItems(f32));
        for (values.constItems(f32), 0..) |value, i| {
            const scale_index = (i / (256 * 3)) * 24 + ((i / 3) % 256) / 32 * 3 + i % 3;
            const normalized = std.math.clamp(host[i] / expected_scales[scale_index], -fp8_max, fp8_max);
            const expected = if (dtype == .f8e4m3fn) zml.floats.Float8E4M3FN.fromF32(normalized).toF32() else zml.floats.Float8E4M3FNUZ.fromF32(normalized).toF32();
            try std.testing.expectEqual(expected, value);
        }
    }
}

test "QuantizedTensor.validate requires logical dtypes, grids and layouts" {
    const input: QuantizedTensor = .{
        .values = .init(.{ .m = 2, .k = 64 }, .f8e4m3fn),
        .scales = .{ .scaling = Scheme.mxfp8.scaling, .values = .init(.{ .m = 2, .k = 2 }, .f8e8m0) },
    };
    try input.validate();
    var invalid = input;
    invalid.values = .fromShape(input.values.shape().withDtype(.u8));
    try std.testing.expectError(error.InvalidValueDtype, invalid.validate());

    invalid = input;
    invalid.scales.values = .init(.{ .m = 2, .k = 2 }, .u8);
    try std.testing.expectError(error.InvalidScaleDtype, invalid.validate());
    invalid.scales.values = .init(.{ .m = 2, .k = 4 }, .f8e8m0);
    try std.testing.expectError(error.InvalidScaleShape, invalid.validate());
    invalid.scales.values = .init(.{4}, .f8e8m0);
    try std.testing.expectError(error.InvalidScaleShape, invalid.validate());
    invalid.scales.values = .init(.{ .m = 2, .k = 64 }, .f8e8m0);
    try std.testing.expectError(error.InvalidScaleShape, invalid.validate());

    invalid = input;
    invalid.values = .init(.{ 64, 64 }, .f8e4m3fn);
    try std.testing.expectError(error.InvalidScaleShape, invalid.validate());
    invalid.values = .init(.{ 64, 64, 64 }, .f8e4m3fn);
    invalid.scales.scaling = .{ .tile = 32 };
    invalid.scales.values = .init(.{ 2, 2, 2 }, .f8e8m0);
    try std.testing.expectError(error.InvalidScaleShape, invalid.validate());

    invalid = input;
    invalid.scales.values = .init(.{}, .f8e8m0);
    for ([_]Scaling{ .{ .axis = 0 }, .{ .tile = 0 } }) |scaling| {
        invalid.scales.scaling = scaling;
        try std.testing.expectError(error.InvalidGroupSize, invalid.validate());
    }
    invalid = input;
    invalid.scales.swizzled = true;
    try std.testing.expectError(error.UnsupportedScaleLayout, invalid.validate());
}

test "dequantize applies logical E8M0 tile scales" {
    const Local = struct {
        fn forward() Tensor {
            const shape: Shape = .init(.{ .n = 64, .k = 64 }, .f8e4m3fn);
            const values = Tensor.scalar(2, .f8e4m3fn).broad(shape);
            const native = Tensor.arange(.{ .end = 4 }, .i32).addConstant(126).convert(.u8)
                .reshape(.{ .n = 2, .k = 2 }).bitCast(.f8e8m0);
            return dequantize(.{ .values = values, .scales = .{
                .scaling = .{ .tile = 32 },
                .values = native,
            } }, .f32);
        }
    };
    const platform = zml.testing.env();
    {
        var exe = try platform.compileFn(std.testing.allocator, std.testing.io, Local.forward, .{}, .{});
        defer exe.deinit();
        var output = try exe.eval(std.testing.allocator, std.testing.io, {});
        defer output.deinit();
        const actual = try output.getValue([64][64]f32, std.testing.io);
        const expected = [_]f32{ 1, 2, 4, 8 };
        for (actual, 0..) |row, n| for (row, 0..) |value, k| {
            try std.testing.expectEqual(expected[(n / 32) * 2 + k / 32], value);
        };
    }
}

test "dequantize expands tensor, axis and square tile grids including partial groups" {
    const Case = struct { scaling: Scaling, scales: Shape, n_block: usize, k_block: usize };
    const Local = struct {
        fn forward(c: Case) Tensor {
            const shape: Shape = .init(.{ .b = 2, .n = 5, .k = 7 }, .f8e4m3fn);
            const values = Tensor.arange(.{ .end = 70 }, .i32).remainder(.scalar(3, .i32)).addConstant(-1).convert(.f8e4m3fn).reshape(shape);
            const q: Scales = .{
                .scaling = c.scaling,
                .values = Tensor.arange(.{ .end = @intCast(c.scales.count()) }, .f32).addConstant(1).reshape(c.scales),
                .global = .{ .value = Tensor.scalar(0.5, .f32), .operation = .multiply },
            };
            return dequantize(.{ .values = values, .scales = q }, .bf16).convert(.f32);
        }
    };
    const platform = zml.testing.env();
    for ([_]Case{
        .{ .scaling = .tensor, .scales = .scalar(.f32), .n_block = 5, .k_block = 7 },
        .{ .scaling = .tensor, .scales = .init(.{ .b = 2, .n = 1, .k = 1 }, .f32), .n_block = 5, .k_block = 7 },
        .{ .scaling = .{ .axis = null }, .scales = .init(.{ .b = 2, .n = 1, .k = 7 }, .f32), .n_block = 5, .k_block = 1 },
        .{ .scaling = .{ .axis = 2 }, .scales = .init(.{ .b = 2, .n = 3, .k = 7 }, .f32), .n_block = 2, .k_block = 1 },
        .{ .scaling = .{ .tile = 3 }, .scales = .init(.{ .b = 2, .n = 2, .k = 3 }, .f32), .n_block = 3, .k_block = 3 },
    }) |c| {
        var exe = try platform.compileFn(std.testing.allocator, std.testing.io, Local.forward, .{c}, .{});
        defer exe.deinit();
        try zml.testing.expectEqualShapes(.init(.{ .b = 2, .n = 5, .k = 7 }, .f32), exe.output_shapes[0]);
        var output = try exe.eval(std.testing.allocator, std.testing.io, {});
        defer output.deinit();
        const actual = try output.getValue([70]f32, std.testing.io);
        for (actual, 0..) |value, i| {
            const scale_index = if (c.scales.rank() == 0) 0 else ((i / 35) * @as(usize, @intCast(c.scales.dim(.n))) + (i / 7 % 5) / c.n_block) * @as(usize, @intCast(c.scales.dim(.k))) + (i % 7) / c.k_block;
            const source: f32 = @floatFromInt(@as(i32, @intCast(i % 3)) - 1);
            try std.testing.expectEqual(source * @as(f32, @floatFromInt(scale_index + 1)) / 2, value);
        }
    }
}

test "scale coverage distinguishes identical partial grids" {
    const Local = struct {
        fn forward(block: u32) Tensor {
            const values = Tensor.scalar(1, .f8e4m3fn).broad(.init(.{ .k = 65 }, .f8e4m3fn));
            return dequantize(.{ .values = values, .scales = .{
                .values = Tensor.arange(.{ .end = 3 }, .f32).addConstant(1).withTags(.{.k}),
                .scaling = .{ .axis = block },
            } }, .f32);
        }
    };
    const platform = zml.testing.env();
    for ([_]u32{ 24, 32 }) |block| {
        var exe = try platform.compileFn(std.testing.allocator, std.testing.io, Local.forward, .{block}, .{});
        defer exe.deinit();
        var output = try exe.eval(std.testing.allocator, std.testing.io, {});
        defer output.deinit();
        for (try output.getValue([65]f32, std.testing.io), 0..) |value, i| {
            try std.testing.expectEqual(@as(f32, @floatFromInt(i / block + 1)), value);
        }
    }
}

test "Scales.reuseBuffer reuses scales while preserving coverage and scale polarity" {
    const Local = struct {
        fn forward(previous: Scales) Scales {
            var next = previous;
            next.values = previous.values.addConstant(1);
            if (previous.global) |scale| next.global = .{ .value = scale.value.addConstant(2), .operation = scale.operation };
            next.scaling = .{ .axis = null };
            const result = next.reuseBuffer(previous);
            std.debug.assert(result.scaling.eql(next.scaling));
            if (result.global) |scale| std.debug.assert(scale.operation == .divide);
            return result;
        }
    };
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    for ([_]bool{ false, true }) |with_global| {
        const previous: Scales = .{
            .scaling = .{ .axis = 32 },
            .values = .init(.{ .m = 2, .k = 1 }, .f32),
            .global = if (with_global) .{ .value = .init(.{}, .f32), .operation = .divide } else null,
        };
        var exe = try platform.compileFn(allocator, io, Local.forward, .{previous}, .{});
        defer exe.deinit();
        var buffers: zml.Bufferized(Scales) = .{
            .values = try .fromBytes(io, platform, previous.values.shape(), std.mem.asBytes(&[2]f32{ 1, 2 })),
            .global = null,
        };
        defer zml.Buffer.deinitAll(Scales, &buffers);
        if (with_global) buffers.global = .{ .value = try .fromBytes(io, platform, .scalar(.f32), std.mem.asBytes(&@as(f32, 2))) };
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        for (0..2) |step| {
            const scales_ptr = buffers.values.opaqueDevicePtr(0);
            const global_ptr = if (buffers.global) |scale| scale.value.opaqueDevicePtr(0) else null;
            runner.run(io, .{buffers}, .{&buffers}, .{ .wait = true });
            try std.testing.expectEqual(scales_ptr, buffers.values.opaqueDevicePtr(0));
            const expected: f32 = @floatFromInt(step + 2);
            try std.testing.expectEqual([2]f32{ expected, expected + 1 }, try buffers.values.getValue([2]f32, io));
            if (buffers.global) |scale| {
                try std.testing.expectEqual(global_ptr.?, scale.value.opaqueDevicePtr(0));
                try std.testing.expectEqual(expected * 2, try scale.value.getValue(f32, io));
            }
        }
    }
}
