//! Quantization formats, metadata, and tensor quantization.
const std = @import("std");

const stdx = @import("stdx");

const zml = @import("zml.zig");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const quantization_mod = @This();

pub const nvfp4_block_size = 16;
pub const mx_block_size = 32;

pub const Quantization = struct {
    scheme: quantization_mod.Scheme,
    scales: Tensor,
    global_scale: ?GlobalScale = null,
    /// Scales use Blackwell's 128x4 blocked MMA layout.
    swizzled_scales: bool = false,

    pub const ValidationError = error{
        InvalidValueDtype,
        InvalidScaleDtype,
        InvalidGroupSize,
        InvalidScaleShape,
        UnsupportedScaleLayout,
    };

    /// Validate unpacked values and a plain scale grid in the values' axis order.
    /// A single scale may broadcast over all values. Otherwise, axis scaling
    /// reduces at most one dimension and tile scaling at most two.
    pub fn validatePlain(self: Quantization, values: Shape) ValidationError!void {
        if (self.swizzled_scales) return error.UnsupportedScaleLayout;
        if (values.dtype() != self.scheme.values) return error.InvalidValueDtype;
        if (!self.acceptsScaleDtype()) return error.InvalidScaleDtype;
        switch (self.scheme.scaling) {
            .axis => |size| if (size == 0) return error.InvalidGroupSize,
            .tile => |size| if (size == 0) return error.InvalidGroupSize,
            .tensor => {},
        }
        if (self.scales.count() == 1) return;
        if (self.scales.rank() != values.rank()) return error.InvalidScaleShape;
        var reduced_axes: usize = 0;
        var grouped_axes: usize = 0;
        for (0..values.rank()) |axis| {
            const block: i64 = switch (self.scheme.scaling) {
                .tensor => values.dim(axis),
                .axis => |size| size orelse values.dim(axis),
                .tile => |size| size,
            };
            if (block <= 0) return error.InvalidGroupSize;
            const groups = std.math.divCeil(i64, values.dim(axis), block) catch return error.InvalidScaleShape;
            if (self.scales.dim(axis) == groups) grouped_axes += 1;
            if (self.scales.dim(axis) == values.dim(axis)) continue;
            if (self.scales.dim(axis) != groups) return error.InvalidScaleShape;
            reduced_axes += 1;
        }
        switch (self.scheme.scaling) {
            .axis => if (reduced_axes > 1 or grouped_axes < 1) return error.InvalidScaleShape,
            .tile => if (reduced_axes > 2 or grouped_axes < 2) return error.InvalidScaleShape,
            .tensor => if (grouped_axes < 1) return error.InvalidScaleShape,
        }
    }

    /// Decode byte-backed E8M0 scales without changing their physical layout.
    pub fn logicalScales(self: Quantization) Tensor {
        stdx.debug.assert(self.acceptsScaleDtype(), "expected {s} scales, got {s}", .{ @tagName(self.scheme.scales), @tagName(self.scales.dtype()) });
        return if (self.scheme.scales == .f8e8m0 and self.scales.dtype() == .u8) self.scales.bitCast(.f8e8m0) else self.scales;
    }

    fn acceptsScaleDtype(self: Quantization) bool {
        return self.scales.dtype() == self.scheme.scales or (self.scheme.scales == .f8e8m0 and self.scales.dtype() == .u8);
    }

    pub fn reuseBuffer(self: Quantization, other: Quantization) Quantization {
        var result = self;
        result.scales = self.scales.reuseBuffer(other.scales);
        if (result.global_scale) |*scale| {
            if (other.global_scale) |previous| scale.value = scale.value.reuseBuffer(previous.value);
        }
        return result;
    }

    /// A per-tensor scale and the polarity its producer wrote it in
    pub const GlobalScale = struct {
        value: Tensor,
        operation: Operation,

        pub const Operation = enum { multiply, divide };

        pub fn multiplier(value: Tensor) GlobalScale {
            return .{ .value = value, .operation = .multiply };
        }

        pub fn asMultiplier(self: GlobalScale) Tensor {
            const s = self.value.convert(.f32);
            const flat = if (s.shape().count() == 1) s.asScalar() else s; // essentially normalization

            return switch (self.operation) {
                .multiply => flat,
                .divide => Tensor.scalar(1.0, .f32).broad(flat.shape()).div(flat),
            };
        }
    };

    pub const Scheme = quantization_mod.Scheme;
};

/// Logical value encoding and scale coverage.
/// Consumers bind scaling to tensor axes and handle physical storage layouts.
pub const Scheme = struct {
    values: DataType,
    scales: DataType,
    scaling: Scaling,

    pub const Scaling = union(enum) {
        /// One scale over the selected axes. Unselected axes remain independent.
        tensor,
        /// Positive group size along one selected axis; null covers that entire axis.
        axis: ?u32,
        /// Positive side length of a square tile over two selected axes.
        tile: u32,
    };

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

/// Quantize groups along the selected axis to exactly the requested logical
/// format. Unsupported formats are reported without choosing another encoding.
/// NVFP4 preserves the supplied global scale and its multiply/divide convention.
pub fn quantize(x: Tensor, scheme: Scheme, axis: anytype, global_scale: ?Quantization.GlobalScale) error{UnsupportedScheme}!struct { Tensor, Quantization } {
    if (Scheme.eql(scheme, Scheme.nvfp4)) return quantizeNvfp4(x, global_scale, axis);
    if (global_scale != null) return error.UnsupportedScheme;
    if (Scheme.eql(scheme, Scheme.mxfp4)) return quantizeMxfp4(x, axis);
    if (scheme.scaling != .axis) return error.UnsupportedScheme;
    if ((scheme.values != .f8e4m3fn and scheme.values != .f8e4m3fnuz) or
        (scheme.scales != .f32 and scheme.scales != .bf16 and scheme.scales != .f8e8m0)) return error.UnsupportedScheme;
    return quantizeBlockFp8(x, axis, scheme);
}

/// Reconstruct logical values in FP32, then convert to dtype.
/// Scales use the same axis order as values, with one entry per group or tile;
/// a scalar scale broadcasts over the entire tensor. Unpack values and unswizzle
/// scales first; byte-backed E8M0 scales are decoded here.
pub fn dequantize(values: Tensor, q: Quantization, dtype: DataType) Tensor {
    q.validatePlain(values.shape()) catch |err| stdx.debug.panic("dequantize: {s} for scheme {any}, values {f}, scales {f}", .{ @errorName(err), q.scheme, values, q.scales });
    var scales = q.logicalScales().convert(.f32);
    if (scales.count() == 1) {
        scales = scales.asScalar();
    } else {
        for (0..values.rank()) |axis| {
            if (scales.dim(axis) == values.dim(axis)) continue;
            const block: i64 = switch (q.scheme.scaling) {
                .tensor => values.dim(axis),
                .axis => |size| size orelse values.dim(axis),
                .tile => |size| size,
            };
            const groups = std.math.divCeil(i64, values.dim(axis), block) catch unreachable;
            const expanded = scales.shape().insert(axis + 1, .{block});
            scales = scales.broadcast(expanded, Shape.range(expanded.rank(), .i64).remove(axis + 1).dims())
                .reshape(scales.shape().setDim(axis, groups * block))
                .slice(axis, .{ .end = values.dim(axis) });
        }
    }
    var result = values.convert(.f32).mul(scales.broad(values.shape().withDtype(.f32)));
    if (q.global_scale) |scale| result = result.mul(scale.asMultiplier().broad(result.shape()));
    return result.convert(dtype);
}

fn quantizeMxfp4(x: Tensor, axis: anytype) struct { Tensor, Quantization } {
    stdx.debug.assert(@mod(x.dim(axis), mx_block_size) == 0, "MXFP4 activation width must be divisible by {}, got {f}", .{ mx_block_size, x.shape() });

    const grouped = x.convert(.f32).splitAxis(axis, .{ .mx_ks = -1, .mx_block = mx_block_size });
    const raw_scale = grouped.abs().max(.mx_block).maximum(.scalar(6 * 0x1p-126, .f32)).scale(1.0 / 6.0);
    const scales = ceilPowerOfTwo(raw_scale);

    return .{
        grouped.div(scales.broad(grouped.shape()))
            .clamp(.scalar(-6, .f32), .scalar(6, .f32))
            .convert(.f4e2m1)
            .reshape(x.shape().withDtype(.f4e2m1)),
        .{
            .scheme = Scheme.mxfp4,
            .scales = scales.reshape(x.shape().setDim(axis, @divExact(x.dim(axis), mx_block_size)).withDtype(.f32)).convert(.f8e8m0),
        },
    };
}

fn ceilPowerOfTwo(raw_scale: Tensor) Tensor {
    const bits = raw_scale.bitCast(.u32);
    return bits.addConstant(0x7fffff).logical(.AND, .scalar(0xff800000, .u32)).bitCast(.f32);
}

/// Quantize activation blocks, preserving the input's axes.
/// E8M0 scales round upward to a power of two to avoid overflowing FP8 values.
fn quantizeBlockFp8(x: Tensor, axis: anytype, scheme: Scheme) struct { Tensor, Quantization } {
    const block_size: i64 = scheme.scaling.axis orelse x.dim(axis);
    const dtype = scheme.values;
    const scale_dtype = scheme.scales;
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
        .f32, .bf16 => grouped.abs()
            .maximum(.scalar(1e-10, .f32))
            .scale(1.0 / fp8_max)
            .max(.fp8_block),
        .f8e8m0 => blk: {
            const raw_scale = grouped.abs().maximum(.scalar(1e-4, .f32)).scale(1.0 / fp8_max).max(.fp8_block);
            break :blk ceilPowerOfTwo(raw_scale);
        },
        else => stdx.debug.panic("expected F32, BF16 or E8M0 scale dtype, got {s}", .{@tagName(scale_dtype)}),
    };
    const stored_scales = scales.convert(scale_dtype);
    const values = grouped.div(stored_scales.convert(.f32).broad(grouped.shape()))
        .clamp(.scalar(-fp8_max, .f32), .scalar(fp8_max, .f32))
        .convert(dtype)
        .reshape(x.shape().withDtype(dtype));
    return .{
        values,
        .{
            .scheme = scheme,
            .scales = stored_scales.reshape(x.shape().setDim(axis, @divExact(x.dim(axis), block_size)).withDtype(scale_dtype)),
        },
    };
}

fn quantizeNvfp4(x: Tensor, global_scale: ?Quantization.GlobalScale, axis: anytype) struct { Tensor, Quantization } {
    const value_max = 6.0;
    const scale_min = 0x1p-9; // Smallest nonzero E4M3 scale, including subnormals.
    const scale_max = DataType.f8e4m3fn.maxValue().as(f32);

    stdx.debug.assert(@rem(x.dim(axis), nvfp4_block_size) == 0, "quantizeNvfp4 expects {any} to be a multiple of {}, got {f}", .{ axis, nvfp4_block_size, x.shape() });

    const xf = x.convert(.f32);
    const scaled = if (global_scale) |scale|
        xf.div(scale.asMultiplier().broad(xf.shape()))
    else
        xf;
    const grouped = scaled.splitAxis(axis, .{ .sc = -1, .blk = nvfp4_block_size });

    const amax = grouped.abs()
        .max(.blk)
        .maximum(.scalar(value_max * scale_min, .f32))
        .scale(1.0 / value_max)
        .minimum(.scalar(scale_max, .f32));

    const scales = amax.convert(.f8e4m3fn);

    const divisor = scales.convert(.f32).broad(grouped.shape());

    const values = grouped.div(divisor)
        .clamp(.scalar(-value_max, .f32), .scalar(value_max, .f32))
        .convert(.f4e2m1)
        .reshape(x.shape().withDtype(.f4e2m1));

    return .{
        values,
        .{
            .scheme = Scheme.nvfp4,
            .scales = scales.reshape(x.shape().setDim(axis, @divExact(x.dim(axis), nvfp4_block_size)).withDtype(.f8e4m3fn)),
            .global_scale = global_scale,
        },
    };
}

test "plain quantization validation rejects mismatched encodings, grids and layouts" {
    const values: Shape = .init(.{ .m = 2, .k = 64 }, .f8e4m3fn);
    const q: Quantization = .{ .scheme = Scheme.mxfp8, .scales = .init(.{ .m = 2, .k = 2 }, .u8) };
    try q.validatePlain(values);
    try std.testing.expectError(error.InvalidValueDtype, q.validatePlain(values.withDtype(.f8e4m3fnuz)));

    var invalid = q;
    invalid.scales = .init(.{ .m = 2, .k = 2 }, .f32);
    try std.testing.expectError(error.InvalidScaleDtype, invalid.validatePlain(values));
    invalid.scales = .init(.{ .m = 2, .k = 4 }, .u8);
    try std.testing.expectError(error.InvalidScaleShape, invalid.validatePlain(values));
    invalid.scales = .init(.{4}, .u8);
    try std.testing.expectError(error.InvalidScaleShape, invalid.validatePlain(values));
    invalid.scales = .init(.{ .m = 2, .k = 64 }, .u8);
    try std.testing.expectError(error.InvalidScaleShape, invalid.validatePlain(values));
    try std.testing.expectError(error.InvalidScaleShape, q.validatePlain(.init(.{ 64, 64 }, .f8e4m3fn)));

    invalid = q;
    invalid.scheme.scaling = .{ .tile = 32 };
    invalid.scales = .init(.{ 2, 2, 2 }, .u8);
    try std.testing.expectError(error.InvalidScaleShape, invalid.validatePlain(.init(.{ 64, 64, 64 }, .f8e4m3fn)));

    invalid = q;
    invalid.scales = .init(.{}, .u8);
    for ([_]Scheme.Scaling{ .{ .axis = 0 }, .{ .tile = 0 } }) |scaling| {
        invalid.scheme.scaling = scaling;
        try std.testing.expectError(error.InvalidGroupSize, invalid.validatePlain(values));
    }
    invalid = q;
    invalid.swizzled_scales = true;
    try std.testing.expectError(error.UnsupportedScaleLayout, invalid.validatePlain(values));
}

test "dequantize decodes byte-backed E8M0 tile scales" {
    const Local = struct {
        fn forward(bytes: bool) Tensor {
            const shape: Shape = .init(.{ .n = 64, .k = 64 }, .f8e4m3fn);
            const values = Tensor.scalar(2, .f8e4m3fn).broad(shape);
            const native = Tensor.arange(.{ .end = 4 }, .i32).addConstant(126).convert(.u8)
                .reshape(.{ .n = 2, .k = 2 }).bitCast(.f8e8m0);
            return dequantize(values, .{
                .scheme = .{ .values = .f8e4m3fn, .scales = .f8e8m0, .scaling = .{ .tile = 32 } },
                .scales = if (bytes) native.bitCast(.u8) else native,
            }, .f32);
        }
    };
    const platform = zml.testing.env();
    for ([_]bool{ false, true }) |bytes| {
        var exe = try platform.compileFn(std.testing.allocator, std.testing.io, Local.forward, .{bytes}, .{});
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

test "quantize rejects unsupported requested formats" {
    const x: Tensor = .init(.{ .row = 2, .feature = 32 }, .bf16);
    for ([_]Scheme{
        .{ .values = .f8e4m3fn, .scales = .f8e8m0, .scaling = .{ .tile = 32 } },
        .{ .values = .f4e2m1, .scales = .f8e4m3fn, .scaling = .{ .axis = 8 } },
        .{ .values = .f8e4m3fn, .scales = .f16, .scaling = .{ .axis = 32 } },
        .{ .values = .i8, .scales = .f32, .scaling = .{ .axis = 32 } },
    }) |scheme| {
        try std.testing.expectError(error.UnsupportedScheme, quantize(x, scheme, .feature, null));
    }
}

test "NVFP4 quantize preserves global scale polarity and reconstructs values" {
    const Local = struct {
        fn forward(operation: Quantization.GlobalScale.Operation) Tensor {
            const scale: Quantization.GlobalScale = switch (operation) {
                .multiply => .multiplier(Tensor.scalar(2, .f32)),
                .divide => .{ .value = Tensor.scalar(0.5, .f32), .operation = .divide },
            };
            const x = Tensor.arange(.{ .end = 32 }, .f32).remainder(Tensor.scalar(4, .f32)).scale(1.5).withTags(.{.k});
            const values, const q = quantize(x, Scheme.nvfp4, .k, scale) catch unreachable;
            std.debug.assert(q.global_scale.?.operation == operation);
            std.debug.assert(q.global_scale.?.value.id == scale.value.id);
            return dequantize(values, q, .f32);
        }
    };
    const platform = zml.testing.env();
    for ([_]Quantization.GlobalScale.Operation{ .multiply, .divide }) |operation| {
        var exe = try platform.compileFn(std.testing.allocator, std.testing.io, Local.forward, .{operation}, .{});
        defer exe.deinit();
        var result = try exe.eval(std.testing.allocator, std.testing.io, {});
        defer result.deinit();
        for (try result.getValue([32]f32, std.testing.io), 0..) |value, i| {
            try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i % 4)) * 1.5, value, 1e-5);
        }
    }
}

test "NVFP4 uses FP32 arithmetic, subnormal scales and finite saturation" {
    const maxima = [_]f32{ 0, 1e-6, 0.001, 6.0 / 512.0, 0.05, 1.75, 3.25, 2688, 1e5 };
    const expected_scales = [_]f32{ 0x1p-9, 0x1p-9, 0x1p-9, 0x1p-9, 0x1p-7, 0.28125, 0.5625, 448, 448 };
    const Local = struct {
        const Outputs = struct { values: Tensor, scales: Tensor, decoded: Tensor };
        fn forward(x: Tensor, dtype: DataType) Outputs {
            const values, const q = quantize(x.convert(dtype), Scheme.nvfp4, .k, null) catch unreachable;
            return .{ .values = values.convert(.f32), .scales = q.scales.convert(.f32), .decoded = dequantize(values, q, .f32) };
        }

        fn roundValue(x: f32) f32 {
            const grid = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
            var best: usize = 0;
            for (grid, 0..) |value, i| {
                const distance = @abs(@abs(x) - value);
                const previous = @abs(@abs(x) - grid[best]);
                if (distance < previous or (distance == previous and i % 2 == 0)) best = i;
            }
            return if (x < 0) -grid[best] else grid[best];
        }
    };
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const x: Tensor = .init(.{ .b = maxima.len, .k = nvfp4_block_size }, .f32);
    var host: [maxima.len * nvfp4_block_size]f32 = undefined;
    for (&host, 0..) |*value, i| {
        const fraction = @as(f32, @floatFromInt(@as(i32, @intCast(i % nvfp4_block_size)) - 8)) / 8;
        value.* = maxima[i / nvfp4_block_size] * fraction;
    }
    var buffer = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&host));
    defer buffer.deinit();
    for ([_]DataType{ .f32, .bf16 }) |dtype| {
        var exe = try platform.compileFn(allocator, io, Local.forward, .{ x, dtype }, .{});
        defer exe.deinit();
        var output: zml.Bufferized(Local.Outputs) = undefined;
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        runner.run(io, .{buffer}, .{&output}, .{ .wait = true });
        defer zml.Buffer.deinitAll(Local.Outputs, &output);
        const values = try output.values.getValue(@TypeOf(host), io);
        const scales = try output.scales.getValue([maxima.len]f32, io);
        const decoded = try output.decoded.getValue(@TypeOf(host), io);
        for (expected_scales, scales, 0..) |expected_scale, scale, group| {
            try std.testing.expectEqual(expected_scale, scale);
            for (0..nvfp4_block_size) |lane| {
                const i = group * nvfp4_block_size + lane;
                const source = if (dtype == .bf16) zml.floats.BFloat16.fromF32(host[i]).toF32() else host[i];
                const expected = Local.roundValue(source / expected_scale);
                try std.testing.expectEqual(expected, values[i]);
                try std.testing.expectEqual(expected * expected_scale, decoded[i]);
            }
        }
    }
}

test "dequantize expands tensor, axis and square tile grids including partial groups" {
    const Case = struct { scaling: Scheme.Scaling, scales: Shape, n_block: usize, k_block: usize };
    const Local = struct {
        fn forward(c: Case) Tensor {
            const shape: Shape = .init(.{ .b = 2, .n = 5, .k = 7 }, .f8e4m3fn);
            const values = Tensor.arange(.{ .end = 70 }, .i32).remainder(.scalar(3, .i32)).addConstant(-1).convert(.f8e4m3fn).reshape(shape);
            const q: Quantization = .{
                .scheme = .{ .values = .f8e4m3fn, .scales = .f32, .scaling = c.scaling },
                .scales = Tensor.arange(.{ .end = @intCast(c.scales.count()) }, .f32).addConstant(1).reshape(c.scales),
                .global_scale = .{ .value = Tensor.scalar(2, .f32), .operation = .divide },
            };
            return dequantize(values, q, .bf16).convert(.f32);
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

test "Quantization.reuseBuffer reuses scales while preserving scheme and polarity" {
    const Local = struct {
        fn forward(previous: Quantization) Quantization {
            var next = previous;
            next.scales = previous.scales.addConstant(1);
            if (previous.global_scale) |scale| next.global_scale = .{ .value = scale.value.addConstant(2), .operation = .divide };
            next.scheme.scaling = .{ .axis = null };
            const result = next.reuseBuffer(previous);
            std.debug.assert(Scheme.eql(result.scheme, next.scheme));
            if (result.global_scale) |scale| std.debug.assert(scale.operation == .divide);
            return result;
        }
    };
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    for ([_]bool{ false, true }) |with_global| {
        const previous: Quantization = .{
            .scheme = .{ .values = .f8e4m3fn, .scales = .f32, .scaling = .{ .axis = 32 } },
            .scales = .init(.{ .m = 2, .k = 1 }, .f32),
            .global_scale = if (with_global) .multiplier(.init(.{}, .f32)) else null,
        };
        var exe = try platform.compileFn(allocator, io, Local.forward, .{previous}, .{});
        defer exe.deinit();
        var buffers: zml.Bufferized(Quantization) = .{
            .scales = try .fromBytes(io, platform, previous.scales.shape(), std.mem.asBytes(&[2]f32{ 1, 2 })),
            .global_scale = null,
        };
        defer zml.Buffer.deinitAll(Quantization, &buffers);
        if (with_global) buffers.global_scale = .{ .value = try .fromBytes(io, platform, .scalar(.f32), std.mem.asBytes(&@as(f32, 2))) };
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        for (0..2) |step| {
            const scales_ptr = buffers.scales.opaqueDevicePtr(0);
            const global_ptr = if (buffers.global_scale) |scale| scale.value.opaqueDevicePtr(0) else null;
            runner.run(io, .{buffers}, .{&buffers}, .{ .wait = true });
            try std.testing.expectEqual(scales_ptr, buffers.scales.opaqueDevicePtr(0));
            const expected: f32 = @floatFromInt(step + 2);
            try std.testing.expectEqual([2]f32{ expected, expected + 1 }, try buffers.scales.getValue([2]f32, io));
            if (buffers.global_scale) |scale| {
                try std.testing.expectEqual(global_ptr.?, scale.value.opaqueDevicePtr(0));
                try std.testing.expectEqual(expected * 2, try scale.value.getValue(f32, io));
            }
        }
    }
}

test "requested cache schemes preserve encodings and reconstruct groups along a nonminor axis" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    for ([_]Scheme{
        Scheme.mxfp8,
        Scheme.nvfp4,
        Scheme.mxfp4,
        .{ .values = .f8e4m3fn, .scales = .bf16, .scaling = .{ .axis = 32 } },
        .{ .values = .f8e4m3fn, .scales = .f32, .scaling = .{ .axis = null } },
    }) |scheme| {
        const Local = struct {
            const Outputs = struct { values: Tensor, scales: Tensor, decoded: Tensor };
            fn forward(x: Tensor, requested: Scheme) Outputs {
                const global_scale: ?Quantization.GlobalScale = if (Scheme.eql(requested, Scheme.nvfp4)) .multiplier(Tensor.scalar(2, .f32)) else null;
                const quantized, const q = quantize(x, requested, .feature, global_scale) catch unreachable;
                std.debug.assert(Scheme.eql(q.scheme, requested));
                std.debug.assert(quantized.dtype() == requested.values and q.scales.dtype() == requested.scales);
                var values = quantized.convert(.f32);
                if (q.global_scale) |scale| values = values.mul(scale.asMultiplier());
                return .{ .values = values, .scales = q.scales.convert(.f32), .decoded = dequantize(quantized, q, .f32) };
            }
        };
        const x: Tensor = .init(.{ .token = 2, .feature = 64, .head = 3 }, .f32);
        var exe = try platform.compileFn(allocator, io, Local.forward, .{ x, scheme }, .{});
        defer exe.deinit();
        const block: usize = scheme.scaling.axis orelse 64;
        try zml.testing.expectEqualShapes(x.shape(), exe.output_shapes[0]);
        try zml.testing.expectEqualShapes(.init(.{ .token = 2, .feature = @as(i64, @intCast(64 / block)), .head = 3 }, .f32), exe.output_shapes[1]);
        var host: [2 * 64 * 3]f32 = undefined;
        for (&host, 0..) |*value, i| {
            const head_value: f32 = switch (i % 3) {
                0 => 0.5,
                1 => 1.0,
                else => -2.0,
            };
            value.* = if (i < 64 * 3) head_value else -head_value / 2;
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
        var decoded = try output.decoded.toSliceAlloc(allocator, io);
        defer decoded.free(allocator);
        for (values.constItems(f32), 0..) |value, i| {
            const scale_index = (i / (64 * 3)) * (64 / block) * 3 + ((i / 3) % 64) / block * 3 + i % 3;
            try std.testing.expectApproxEqAbs(host[i], value * scales.constItems(f32)[scale_index], 0.07);
            try std.testing.expectEqual(value * scales.constItems(f32)[scale_index], decoded.constItems(f32)[i]);
        }
    }
}

test "block FP8 quantization preserves axes and reconstructs constant blocks in FN and FNUZ" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    inline for (.{ DataType.f8e4m3fn, DataType.f8e4m3fnuz }) |dtype| {
        const Local = struct {
            const Outputs = struct { values: Tensor, scales: Tensor };
            fn forward(x: Tensor) Outputs {
                const input, const q = quantize(x, .{ .values = dtype, .scales = .f32, .scaling = .{ .axis = 128 } }, .k, null) catch unreachable;
                return .{ .values = input.convert(.f32), .scales = q.scales };
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
                const input, const q = quantize(x, .{ .values = dtype, .scales = .f8e8m0, .scaling = .{ .axis = 32 } }, .k, null) catch unreachable;
                return .{ .values = input.convert(.f32), .scales = q.scales.convert(.f32) };
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
