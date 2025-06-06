///! Conversion utilities between different Floating point formats.
const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}

fn allBitsOne(v: anytype) bool {
    return v == std.math.maxInt(@TypeOf(v));
}

fn FloatHelpers(Float: type) type {
    const info = @typeInfo(Float);
    const err_msg = "FloatHelpers expect a packed struct { mantissa: uXX, exponent: uXX, sign: u1}";
    if (info != .@"struct" or info.@"struct".backing_integer == null) {
        @compileError(err_msg);
    }
    comptime {
        for (info.@"struct".fields, &.{ "mantissa", "exponent", "sign" }) |field, expected_name| {
            if (!std.mem.eql(u8, field.name, expected_name))
                @compileError(err_msg);
        }
    }

    return struct {
        const sign_bits: u8 = @typeInfo(@FieldType(Float, "sign")).int.bits;
        const mantissa_bits: u8 = @typeInfo(@FieldType(Float, "mantissa")).int.bits;
        const exponent_bits: u8 = @typeInfo(@FieldType(Float, "exponent")).int.bits;

        pub const zero: Float = .{ .sign = 0, .exponent = 0, .mantissa = 0 };

        pub fn neg(x: Float) Float {
            return .{
                .sign = x.sign ^ 1,
                .exponent = x.exponent,
                .mantissa = x.mantissa,
            };
        }

        /// Lossy conversion from f32, similar to @floatCast
        pub fn fromF32(f: f32) Float {
            const vf32: Float32 = @bitCast(f);
            const exp_bias = comptime expBias();
            const exponent = @as(u16, vf32.exponent) + exp_bias -| FloatHelpers(Float32).expBias();
            const overflow = exponent > std.math.maxInt(@FieldType(Float, "exponent"));
            if (overflow) {
                return if (@hasDecl(Float, "inf")) {
                    return if (vf32.sign == 0) Float.inf else Float.minus_inf;
                } else Float.nan;
            }
            return .{
                .sign = vf32.sign,
                .exponent = @intCast(exponent),
                .mantissa = truncMantissa(vf32.mantissa),
            };
        }

        /// Lossless conversion to f32.
        pub fn toF32(x: Float) f32 {
            var vf32: Float32 = undefined;
            if (@hasDecl(Float, "isInf") and x.isInf()) {
                return if (x.sign == 0) std.math.inf(f32) else -std.math.inf(f32);
            }
            vf32 = .{
                .sign = x.sign,
                .exponent = if (x.exponent == 0) 0 else @intCast(@as(i16, x.exponent) + f32_exp_bias - expBias()),
                .mantissa = f32Mantissa(x),
            };
            return @bitCast(vf32);
        }

        fn truncMantissa(x: anytype) @FieldType(Float, "mantissa") {
            @setRuntimeSafety(false);
            const off = @bitSizeOf(@TypeOf(x)) - mantissa_bits;
            return @intCast(x >> off);
        }

        fn f32Mantissa(x: Float) @FieldType(Float32, "mantissa") {
            @setRuntimeSafety(false);
            const Res = @FieldType(Float32, "mantissa");
            const f32_mantissa_bits = @bitSizeOf(Res);

            return @shlExact(@as(Res, x.mantissa), f32_mantissa_bits - mantissa_bits);
        }

        fn expBias() u8 {
            return std.math.maxInt(std.meta.Int(.unsigned, exponent_bits - 1));
        }

        pub fn format(
            float: Float,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = options;
            if (fmt.len == 1 and fmt[0] == '_') {
                try writer.print("{{ .sign={}, .exp={}, .mantissa={} }}", .{ float.sign, float.exponent, float.mantissa });
            } else {
                try writer.print("{" ++ fmt ++ "}", .{float.toF32()});
            }
        }
    };
}

pub const Float32 = packed struct(u32) {
    mantissa: u23,
    exponent: u8,
    sign: u1,

    const Helpers = FloatHelpers(@This());
    pub const zero = Helpers.zero;
    pub const neg = Helpers.neg;
    pub const fromF32 = Helpers.fromF32;
    pub const toF32 = Helpers.toF32;
    pub const format = Helpers.format;
};

const f32_exp_bias = FloatHelpers(Float32).expBias();

pub const Float64 = packed struct(u64) {
    mantissa: u52,
    exponent: u11,
    sign: u1,

    const Helpers = FloatHelpers(@This());
    pub const zero = Helpers.zero;
    pub const neg = Helpers.neg;
    pub const fromF32 = Helpers.fromF32;
    pub const toF32 = Helpers.toF32;
    pub const format = Helpers.format;
};

pub const Float8E4M3B11FNUZ = packed struct(u8) {
    mantissa: u3,
    exponent: u4,
    sign: u1,

    pub const nan: Float8E4M3B11FNUZ = .{
        .sign = 1,
        .exponent = 0,
        .mantissa = 0,
    };

    pub fn isNan(self: Float8E4M3B11FNUZ) bool {
        return self.sign == 1 and self.exponent == 0 and self.mantissa == 0;
    }

    const Helpers = FloatHelpers(@This());
    pub const zero = Helpers.zero;
    pub const neg = Helpers.neg;
    pub const fromF32 = Helpers.fromF32;
    pub const toF32 = Helpers.toF32;
    pub const format = Helpers.format;
};

pub const Float8E4M3FN = packed struct(u8) {
    mantissa: u3,
    exponent: u4,
    sign: u1,

    pub const nan: Float8E4M3FN = .{ .sign = 0, .exponent = std.math.maxInt(u4), .mantissa = std.math.maxInt(u3) };

    pub fn isNan(self: Float8E4M3FN) bool {
        return allBitsOne(self.exponent) and allBitsOne(self.mantissa);
    }
    const Helpers = FloatHelpers(@This());
    pub const zero = Helpers.zero;
    pub const neg = Helpers.neg;
    pub const fromF32 = Helpers.fromF32;
    pub const toF32 = Helpers.toF32;
    pub const format = Helpers.format;
};

pub const Float8E4M3FNUZ = packed struct(u8) {
    mantissa: u3,
    exponent: u4,
    sign: u1,

    pub const nan: Float8E4M3FNUZ = .{
        .sign = 1,
        .exponent = 0,
        .mantissa = 0,
    };

    pub fn isNan(self: Float8E4M3FNUZ) bool {
        return self.sign == 1 and self.exponent == 0 and self.mantissa == 0;
    }

    const Helpers = FloatHelpers(@This());
    pub const zero = Helpers.zero;
    pub const neg = Helpers.neg;
    pub const fromF32 = Helpers.fromF32;
    pub const toF32 = Helpers.toF32;
    pub const format = Helpers.format;
};

test "Float8E4" {
    const test_case_e4: TestCase = .{
        .lossless = &[_]f32{ 0, 1.0, -2, 1.0 / 64.0, -128 },
        .lossy = &[_]f32{3.02344107628},
    };

    inline for (.{
        Float8E4M3B11FNUZ,
        Float8E4M3FN,
        Float8E4M3FNUZ,
    }) |Float8T| {
        try testCustomFloat(Float8T, test_case_e4);
        try std.testing.expectEqual(0.0, Float8T.fromF32(1.0 / 128.0).toF32());
    }
}

pub const Float8E5M2 = packed struct(u8) {
    mantissa: u2,
    exponent: u5,
    sign: u1,

    pub const nan: Float8E5M2 = .{
        .sign = 0,
        .exponent = std.math.maxInt(u5),
        .mantissa = 1,
    };

    pub fn isNan(self: Float8E5M2) bool {
        return allBitsOne(self.exponent) and self.mantissa != 0;
    }

    pub const minus_inf: Float8E5M2 = .{
        .sign = 1,
        .exponent = std.math.maxInt(u5),
        .mantissa = 0,
    };

    pub const inf: Float8E5M2 = .{
        .sign = 0,
        .exponent = std.math.maxInt(u5),
        .mantissa = 0,
    };

    pub fn isInf(self: Float8E5M2) bool {
        return allBitsOne(self.exponent) and self.mantissa == 0;
    }

    const Helpers = FloatHelpers(@This());
    pub const zero = Helpers.zero;
    pub const neg = Helpers.neg;
    pub const fromF32 = Helpers.fromF32;
    pub const toF32 = Helpers.toF32;
    pub const format = Helpers.format;
};

pub const Float8E5M2FNUZ = packed struct(u8) {
    mantissa: u2,
    exponent: u5,
    sign: u1,

    pub const nan: Float8E5M2FNUZ = .{ .sign = 1, .exponent = 0, .mantissa = 0 };

    pub fn isNan(self: Float8E5M2FNUZ) bool {
        return self.sign == 1 and self.exponent == 0 and self.mantissa == 0;
    }

    const Helpers = FloatHelpers(@This());
    pub const zero = Helpers.zero;
    pub const neg = Helpers.neg;
    pub const fromF32 = Helpers.fromF32;
    pub const toF32 = Helpers.toF32;
    pub const format = Helpers.format;
};

test "Float8E5" {
    const test_case_e5: TestCase = .{
        .lossless = &[_]f32{ 0, 1.0, -2, 1.0 / 128.0, -128 },
        .lossy = &[_]f32{3.02344107628},
    };
    inline for (.{ Float8E5M2, Float8E5M2FNUZ }) |Float8T| {
        try testCustomFloat(Float8T, test_case_e5);
    }
}

pub const BFloat16 = packed struct(u16) {
    mantissa: u7,
    exponent: u8,
    sign: u1,

    pub const nan: BFloat16 = .{ .sign = 0, .exponent = std.math.maxInt(u8), .mantissa = 1 };

    pub fn isNan(self: BFloat16) bool {
        return allBitsOne(self.exponent) and self.mantissa != 0;
    }

    pub const minus_inf: BFloat16 = .{
        .sign = 1,
        .exponent = std.math.maxInt(u8),
        .mantissa = 0,
    };

    pub const inf: BFloat16 = .{
        .sign = 0,
        .exponent = std.math.maxInt(u8),
        .mantissa = 0,
    };

    pub fn isInf(self: BFloat16) bool {
        return allBitsOne(self.exponent) and self.mantissa == 0;
    }

    pub fn toF32(self: BFloat16) f32 {
        // Pad the BF16 with zeros 0
        return @bitCast([2]u16{ 0, @bitCast(self) });
    }

    pub fn fromF32(float32: f32) BFloat16 {
        var int: u32 = @bitCast(float32);
        // Round up if needed.
        int += 0x8000;
        const parts: [2]u16 = @bitCast(int);
        return @bitCast(parts[1]);
    }

    const Helpers = FloatHelpers(@This());
    pub const zero = Helpers.zero;
    pub const neg = Helpers.neg;
    pub const format = Helpers.format;
};

test BFloat16 {
    // From https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Examples
    try std.testing.expectEqual(BFloat16.fromF32(0), BFloat16{ .sign = 0, .exponent = 0, .mantissa = 0 });
    try std.testing.expectEqual(BFloat16.fromF32(-2), BFloat16{ .sign = 1, .exponent = 127 + 1, .mantissa = 0 });
    try std.testing.expectEqual(BFloat16.fromF32(3.02344107628), BFloat16{ .sign = 0, .exponent = 127 + 1, .mantissa = 66 });
    try std.testing.expectEqual(BFloat16.fromF32(1.0 / 128.0), BFloat16{ .sign = 0, .exponent = 127 - 7, .mantissa = 0 });
    try std.testing.expectEqual(std.mem.toBytes(BFloat16.inf.neg()), [_]u8{ 0x80, 0xff });
    try std.testing.expectEqual(BFloat16.inf, BFloat16.fromF32(std.math.inf(f32)));

    try testCustomFloat(BFloat16, .{
        .lossless = &[_]f32{ 0, -2, 1.0 / 128.0, -1e64, std.math.inf(f32) },
        .lossy = &[_]f32{3.02344107628},
    });
}

pub fn floatCast(T: type, x: anytype) T {
    return switch (@TypeOf(x)) {
        f64, f32, f16 => @floatCast(x),
        else => @floatCast(x.toF32()),
    };
}

const TestCase = struct {
    lossless: []const f32,
    lossy: []const f32,
    tolerance: f32 = 1e-2,
};

fn testCustomFloat(FloatT: type, test_case: TestCase) !void {
    for (test_case.lossless) |x| {
        try std.testing.expectEqual(x, FloatT.fromF32(x).toF32());
    }
    for (test_case.lossy) |x| {
        try expectApproxEqRel(f32, x, FloatT.fromF32(x).toF32(), test_case.tolerance);
    }
}

fn expectApproxEqRel(FloatT: type, x: FloatT, y: FloatT, tolerance: FloatT) !void {
    if (!std.math.approxEqRel(f32, x, y, tolerance)) {
        std.log.err("expected ~{d}, got {d}", .{ x, y });
        return error.TestUnexpectedResult;
    }
}
