const std = @import("std");

fn allBitsOne(v: anytype) bool {
    return v == std.math.maxInt(@TypeOf(v));
}

fn FloatType(sign_bits: u1, exponent_bits: u8, mantissa_bits: u8, innerT: type) type {
    return packed struct(std.meta.Int(.unsigned, @intCast(sign_bits + exponent_bits + mantissa_bits))) {
        const Self = @This();

        mantissa: std.meta.Int(.unsigned, @intCast(mantissa_bits)),
        exponent: std.meta.Int(.unsigned, @intCast(exponent_bits)),
        sign: std.meta.Int(.unsigned, @intCast(sign_bits)),

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
            const precision_loss = @bitSizeOf(@TypeOf(vf32.mantissa)) - mantissa_bits;
            return .{
                .sign = vf32.sign,
                .exponent = @intCast(vf32.exponent),
                .mantissa = shr(vf32.mantissa, precision_loss),
            };
        }

        /// Lossless conversion to f32.
        pub fn toF32(self: Self) f32 {
            var vf32: Float32 = undefined;
            const precision_loss = @bitSizeOf(@TypeOf(vf32.mantissa)) - mantissa_bits;
            vf32 = .{
                .sign = self.sign,
                .exponent = self.exponent,
                .mantissa = @shlExact(@as(@TypeOf(vf32.mantissa), self.mantissa), precision_loss),
            };
            return @bitCast(vf32);
        }

        fn truncMantissa(T: type, x: anytype) T {
            const off = @bitSizeOf(@TypeOf(x)) - @bitSizeOf(T);
            return @intCast(x >> off);
        }

        fn shr(x: anytype, comptime off: u8) std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(x)) - off) {
            // @setRuntimeSafety(false);
            return @intCast(x >> off);
        }

        pub fn format(
            self: @This(),
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = options;
            try writer.print("{" ++ fmt ++ "}", .{self.toF32()});
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

    const lossless = [_]f32{ 0, -2, 1.0 / 128.0, -1e64, std.math.inf(f32) };
    for (&lossless) |v| {
        try std.testing.expectEqual(v, BFloat16.fromF32(v).toF32());
    }
    const lossy = [_]f32{3.02344107628};
    for (&lossy) |x| {
        const y = BFloat16.fromF32(x).toF32();
        if (!std.math.approxEqRel(f32, x, y, 1e-2)) {
            std.log.err("expected ~{d}, got {d}", .{ x, y });
            return error.TestUnexpectedResult;
        }
    }
}

pub fn floatCast(T: type, x: anytype) T {
    return switch (@TypeOf(x)) {
        f64, f32, f16 => @floatCast(x),
        else => @floatCast(x.toF32()),
    };
}
