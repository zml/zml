///! Conversion utilities between different Floating point formats.
const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}

fn allBitsOne(v: anytype) bool {
    return v == std.math.maxInt(@TypeOf(v));
}

fn FloatType(sign_bits: u1, exponent_bits: u8, mantissa_bits: u8, innerT: type) type {
    const bit_size = sign_bits + exponent_bits + mantissa_bits;
    if (bit_size % 8 != 0) @compileError("FloatType should have a number of bits divisible by 8");

    return packed struct(std.meta.Int(.unsigned, bit_size)) {
        const Self = @This();

        mantissa: std.meta.Int(.unsigned, mantissa_bits),
        exponent: std.meta.Int(.unsigned, exponent_bits),
        sign: std.meta.Int(.unsigned, sign_bits),

        pub fn zero() Self {
            return .{
                .sign = 0,
                .exponent = 0,
                .mantissa = 0,
            };
        }

        pub fn neg(self: Self) Self {
            return .{
                .sign = self.sign ^ 1,
                .exponent = self.exponent,
                .mantissa = self.mantissa,
            };
        }

        /// Lossy conversion from f32, similar to @floatCast
        pub fn fromF32(f: f32) Self {
            const vf32: Float32 = @bitCast(f);
            const exp_bias = comptime Self.expBias();
            const exponent = @as(u16, vf32.exponent) + exp_bias -| Float32.expBias();
            const overflow = exponent > std.math.maxInt(std.meta.Int(.unsigned, exponent_bits));
            if (overflow) {
                return if (@hasDecl(Self, "inf")) {
                    return if (vf32.sign == 0) Self.inf() else Self.minusInf();
                } else Self.nan();
            }
            return .{
                .sign = vf32.sign,
                .exponent = @intCast(exponent),
                .mantissa = truncMantissa(vf32.mantissa),
            };
        }

        /// Lossless conversion to f32.
        pub fn toF32(self: Self) f32 {
            var vf32: Float32 = undefined;
            if (@hasDecl(Self, "isInf") and self.isInf()) {
                return if (self.sign == 0) std.math.inf(f32) else -std.math.inf(f32);
            }
            vf32 = .{
                .sign = self.sign,
                .exponent = if (self.exponent == 0) 0 else @intCast(@as(i16, self.exponent) + Float32.expBias() - Self.expBias()),
                .mantissa = self.f32Mantissa(),
            };
            return @bitCast(vf32);
        }

        fn truncMantissa(x: anytype) std.meta.FieldType(Self, .mantissa) {
            @setRuntimeSafety(false);
            const off = @bitSizeOf(@TypeOf(x)) - mantissa_bits;
            return @intCast(x >> off);
        }

        fn f32Mantissa(self: Self) std.meta.FieldType(Float32, .mantissa) {
            @setRuntimeSafety(false);
            const f32_mantissa_bits = @bitSizeOf(std.meta.FieldType(Float32, .mantissa));

            const Res = std.meta.FieldType(Float32, .mantissa);
            return @shlExact(@as(Res, self.mantissa), f32_mantissa_bits - mantissa_bits);
        }

        fn expBias() u8 {
            return std.math.maxInt(std.meta.Int(.unsigned, exponent_bits - 1));
        }

        pub fn format(
            self: @This(),
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = options;
            if (fmt.len == 1 and fmt[0] == '_') {
                try writer.print("{{ .sign={}, .exp={}, .mantissa={} }}", .{ self.sign, self.exponent, self.mantissa });
            } else {
                try writer.print("{" ++ fmt ++ "}", .{self.toF32()});
            }
        }

        pub usingnamespace innerT;
    };
}

const Float32 = FloatType(1, 8, 23, struct {});
const Float64 = FloatType(1, 11, 52, struct {});

pub const Float8E4M3B11FNUZ = FloatType(1, 4, 3, struct {
    pub fn nan() Float8E4M3B11FNUZ {
        return .{
            .sign = 1,
            .exponent = 0,
            .mantissa = 0,
        };
    }

    pub fn isNan(self: Float8E4M3B11FNUZ) bool {
        return self.sign == 1 and self.exponent == 0 and self.mantissa == 0;
    }
});

pub const Float8E4M3FN = FloatType(1, 4, 3, struct {
    pub fn nan() Float8E4M3FN {
        return .{
            .sign = 0,
            .exponent = std.math.maxInt(u4),
            .mantissa = std.math.maxInt(u3),
        };
    }

    pub fn isNan(self: Float8E4M3FN) bool {
        return allBitsOne(self.exponent) and allBitsOne(self.mantissa);
    }
});

pub const Float8E4M3FNUZ = FloatType(1, 4, 3, struct {
    pub fn nan() Float8E4M3FNUZ {
        return .{
            .sign = 1,
            .exponent = 0,
            .mantissa = 0,
        };
    }

    pub fn isNan(self: Float8E4M3FNUZ) bool {
        return self.sign == 1 and self.exponent == 0 and self.mantissa == 0;
    }
});

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

pub const Float8E5M2 = FloatType(1, 5, 2, struct {
    pub fn nan() Float8E5M2 {
        return .{
            .sign = 0,
            .exponent = std.math.maxInt(u5),
            .mantissa = 1,
        };
    }

    pub fn isNan(self: Float8E5M2) bool {
        return allBitsOne(self.exponent) and self.mantissa != 0;
    }

    pub fn minusInf() Float8E5M2 {
        return .{
            .sign = 1,
            .exponent = std.math.maxInt(u5),
            .mantissa = 0,
        };
    }

    pub fn inf() Float8E5M2 {
        return .{
            .sign = 0,
            .exponent = std.math.maxInt(u5),
            .mantissa = 0,
        };
    }

    pub fn isInf(self: Float8E5M2) bool {
        return allBitsOne(self.exponent) and self.mantissa == 0;
    }
});

pub const Float8E5M2FNUZ = FloatType(1, 5, 2, struct {
    pub fn nan() Float8E5M2FNUZ {
        return .{
            .sign = 1,
            .exponent = 0,
            .mantissa = 0,
        };
    }

    pub fn isNan(self: Float8E5M2FNUZ) bool {
        return self.sign == 1 and self.exponent == 0 and self.mantissa == 0;
    }
});

test "Float8E5" {
    const test_case_e5: TestCase = .{
        .lossless = &[_]f32{ 0, 1.0, -2, 1.0 / 128.0, -128 },
        .lossy = &[_]f32{3.02344107628},
    };
    inline for (.{ Float8E5M2, Float8E5M2FNUZ }) |Float8T| {
        try testCustomFloat(Float8T, test_case_e5);
    }
}

pub const BFloat16 = FloatType(1, 8, 7, struct {
    pub fn nan() BFloat16 {
        return .{
            .sign = 0,
            .exponent = std.math.maxInt(u8),
            .mantissa = 1,
        };
    }

    pub fn isNan(self: BFloat16) bool {
        return allBitsOne(self.exponent) and self.mantissa != 0;
    }

    pub fn minusInf() BFloat16 {
        return .{
            .sign = 1,
            .exponent = std.math.maxInt(u8),
            .mantissa = 0,
        };
    }

    pub fn inf() BFloat16 {
        return .{
            .sign = 0,
            .exponent = std.math.maxInt(u8),
            .mantissa = 0,
        };
    }

    pub fn isInf(self: BFloat16) bool {
        return allBitsOne(self.exponent) and self.mantissa == 0;
    }
});

test BFloat16 {
    // From https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Examples
    try std.testing.expectEqual(BFloat16.fromF32(0), BFloat16{ .sign = 0, .exponent = 0, .mantissa = 0 });
    try std.testing.expectEqual(BFloat16.fromF32(-2), BFloat16{ .sign = 1, .exponent = 127 + 1, .mantissa = 0 });
    try std.testing.expectEqual(BFloat16.fromF32(3.02344107628), BFloat16{ .sign = 0, .exponent = 127 + 1, .mantissa = 65 });
    try std.testing.expectEqual(BFloat16.fromF32(1.0 / 128.0), BFloat16{ .sign = 0, .exponent = 127 - 7, .mantissa = 0 });
    try std.testing.expectEqual(std.mem.toBytes(BFloat16.inf().neg()), [_]u8{ 0x80, 0xff });
    try std.testing.expectEqual(BFloat16.inf(), BFloat16.fromF32(std.math.inf(f32)));

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
