const std = @import("std");

const floats = @import("floats.zig");

const C64 = std.math.Complex(f32);
const C128 = std.math.Complex(f64);

test {
    std.testing.refAllDecls(@This());
}

pub const DataType = enum(u8) {
    bool,
    // Note: the support of the float8 is a bit spotty, f8e4m3b11fnuz seems to be the most supported one on Cuda.
    f4e2m1,
    f8e3m4,
    f8e4m3,
    f8e4m3b11fnuz,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,
    f8e8m0,
    bf16,
    f16,
    f32,
    f64,
    i2,
    i4,
    i8,
    i16,
    i32,
    i64,
    u2,
    u4,
    u8,
    u16,
    u32,
    u64,
    c64,
    c128,

    pub fn str(self: DataType) [:0]const u8 {
        return switch (self) {
            inline else => |tag| @tagName(tag),
        };
    }

    pub const Class = enum(u8) {
        bool,
        float,
        integer,
        complex,
    };

    pub fn class(self: DataType) Class {
        return switch (self) {
            .bool => .bool,
            .f4e2m1,
            .f8e3m4,
            .f8e4m3,
            .f8e4m3b11fnuz,
            .f8e4m3fn,
            .f8e4m3fnuz,
            .f8e5m2,
            .f8e5m2fnuz,
            .f8e8m0,
            .bf16,
            .f16,
            .f32,
            .f64,
            => .float,
            .i2, .i4, .i8, .i16, .i32, .i64, .u2, .u4, .u8, .u16, .u32, .u64 => .integer,
            .c64, .c128 => .complex,
        };
    }

    pub fn isInteger(self: DataType) bool {
        return self.class() == .integer;
    }

    pub fn isFloat(self: DataType) bool {
        return self.class() == .float;
    }

    pub fn isComplex(self: DataType) bool {
        return self.class() == .complex;
    }

    pub fn fromZigType(comptime T: type) DataType {
        return switch (T) {
            floats.Float4E2M1 => .f4e2m1,
            floats.Float8E3M4 => .f8e3m4,
            floats.Float8E4M3 => .f8e4m3,
            floats.Float8E4M3B11FNUZ => .f8e4m3b11fnuz,
            floats.Float8E4M3FN => .f8e4m3fn,
            floats.Float8E4M3FNUZ => .f8e4m3fnuz,
            floats.Float8E5M2 => .f8e5m2,
            floats.Float8E5M2FNUZ => .f8e5m2fnuz,
            floats.Float8E8M0 => .f8e8m0,
            floats.BFloat16 => .bf16,
            f16 => .f16,
            f32 => .f32,
            f64 => .f64,
            bool => .bool,
            i2 => .i2,
            i4 => .i4,
            i8 => .i8,
            i16 => .i16,
            i32 => .i32,
            i64 => .i64,
            u2 => .u2,
            u4 => .u4,
            u8 => .u8,
            u16 => .u16,
            u32 => .u32,
            u64 => .u64,
            C64 => .c64,
            C128 => .c128,
            else => @compileError("Unsupported Zig type: " ++ @typeName(T)),
        };
    }

    pub fn fromSliceElementType(slice: anytype) DataType {
        const type_info = @typeInfo(@TypeOf(slice));
        if (type_info != .pointer) {
            @compileError("`initFromSlice` expects a slice, got " ++ @tagName(type_info));
        }

        return switch (type_info.pointer.size) {
            .slice, .c, .many => DataType.fromZigType(type_info.pointer.child),
            .one => b: {
                const child_type_info = @typeInfo(type_info.pointer.child);
                break :b DataType.fromZigType(child_type_info.array.child);
            },
        };
    }

    pub fn toZigType(comptime dtype: DataType) type {
        return @FieldType(Value, @tagName(dtype));
    }

    pub fn isSignedInt(dtype: DataType) bool {
        return switch (dtype) {
            .i4, .i8, .i16, .i32, .i64 => true,
            else => false,
        };
    }

    pub fn sizeOf(self: DataType) u16 {
        return switch (self) {
            inline else => |tag| @sizeOf(tag.toZigType()),
        };
    }

    pub fn bitSizeOf(self: DataType) u16 {
        return switch (self) {
            inline else => |tag| @bitSizeOf(tag.toZigType()),
        };
    }

    pub fn alignOf(self: DataType) u29 {
        return switch (self) {
            inline else => |tag| @alignOf(tag.toZigType()),
        };
    }

    /// Try to find a type compatible with both dtype.
    pub fn resolvePeerType(a: DataType, b: DataType) ?DataType {
        if (a == b) {
            return a;
        }

        // only resolve types in the same class
        if (a.class() != b.class()) {
            return null;
        }

        return if (a.sizeOf() >= b.sizeOf()) a else b;
    }

    test resolvePeerType {
        try std.testing.expectEqual(DataType.f16.resolvePeerType(.f16), .f16);
        try std.testing.expectEqual(DataType.f32.resolvePeerType(.f32), .f32);

        try std.testing.expectEqual(DataType.f16.resolvePeerType(.f32), .f32);
        try std.testing.expectEqual(DataType.f32.resolvePeerType(.f16), .f32);

        try std.testing.expectEqual(DataType.f32.resolvePeerType(.f64), .f64);
        try std.testing.expectEqual(DataType.f64.resolvePeerType(.f32), .f64);
        try std.testing.expectEqual(DataType.f32.resolvePeerType(.i32), null);

        try std.testing.expectEqual(DataType.c64.resolvePeerType(.c128), .c128);
        try std.testing.expectEqual(DataType.c128.resolvePeerType(.i32), null);
        try std.testing.expectEqual(DataType.c64.resolvePeerType(.f32), null);
    }

    pub fn zero(dtype: DataType) Value {
        return .init(dtype, 0);
    }

    pub fn one(dtype: DataType) Value {
        return .init(dtype, 1);
    }

    pub fn minValue(dtype: DataType) Value {
        return switch (dtype) {
            .bool => .{ .bool = false },
            inline .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2fnuz, .f8e8m0, .f4e2m1 => |tag| @unionInit(Value, @tagName(tag), @FieldType(Value, @tagName(tag)).min),
            inline .f8e5m2, .f8e3m4, .f8e4m3, .bf16 => |tag| @unionInit(Value, @tagName(tag), @FieldType(Value, @tagName(tag)).minus_inf),
            inline .f16, .f32, .f64 => |tag| @unionInit(Value, @tagName(tag), -std.math.inf(@FieldType(Value, @tagName(tag)))),
            inline .i2, .i4, .i8, .i16, .i32, .i64, .u2, .u4, .u8, .u16, .u32, .u64 => |tag| @unionInit(Value, @tagName(tag), std.math.minInt(@FieldType(Value, @tagName(tag)))),
            inline .c64, .c128 => |tag| @panic("DataType doesn't have a min value: " ++ @tagName(tag)),
        };
    }

    pub fn maxValue(dtype: DataType) Value {
        return switch (dtype) {
            .bool => .{ .bool = true },
            inline .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2fnuz, .f8e8m0, .f4e2m1 => |tag| @unionInit(Value, @tagName(tag), @FieldType(Value, @tagName(tag)).max),
            inline .f8e5m2, .f8e3m4, .f8e4m3, .bf16 => |tag| @unionInit(Value, @tagName(tag), @FieldType(Value, @tagName(tag)).inf),
            inline .f16, .f32, .f64 => |tag| @unionInit(Value, @tagName(tag), std.math.inf(@FieldType(Value, @tagName(tag)))),
            inline .i2, .i4, .i8, .i16, .i32, .i64, .u2, .u4, .u8, .u16, .u32, .u64 => |tag| @unionInit(Value, @tagName(tag), std.math.maxInt(@FieldType(Value, @tagName(tag)))),
            inline .c64, .c128 => |tag| @panic("DataType doesn't have a max value: " ++ @tagName(tag)),
        };
    }

    pub fn constant(dtype: DataType, value: anytype) Value {
        return .init(dtype, value);
    }

    pub const Value = union(DataType) {
        bool: bool,
        f4e2m1: floats.Float4E2M1,
        f8e3m4: floats.Float8E3M4,
        f8e4m3: floats.Float8E4M3,
        f8e4m3b11fnuz: floats.Float8E4M3B11FNUZ,
        f8e4m3fn: floats.Float8E4M3FN,
        f8e4m3fnuz: floats.Float8E4M3FNUZ,
        f8e5m2: floats.Float8E5M2,
        f8e5m2fnuz: floats.Float8E5M2FNUZ,
        f8e8m0: floats.Float8E8M0,
        bf16: floats.BFloat16,
        f16: f16,
        f32: f32,
        f64: f64,
        i2: i2,
        i4: i4,
        i8: i8,
        i16: i16,
        i32: i32,
        i64: i64,
        u2: u2,
        u4: u4,
        u8: u8,
        u16: u16,
        u32: u32,
        u64: u64,
        c64: C64,
        c128: C128,

        /// Creates `Data` from a `value`.
        ///
        /// If the `dtype` and `@TypeOf(value)` are incompatible
        /// or a cast from `value` to `FieldType(dtype)` would
        /// be lossy, a panic occurs.
        pub fn init(dtype_: DataType, value: anytype) Value {
            const T = @TypeOf(value);
            const Ti = @typeInfo(T);

            return switch (dtype_) {
                .bool => switch (Ti) {
                    .bool => .{ .bool = value },
                    .comptime_int, .int, .comptime_float, .float => .{ .bool = value != 0 },
                    else => @panic("Could not create Value of type bool from value of type " ++ @typeName(T)),
                },
                inline .f4e2m1, .f8e3m4, .f8e4m3, .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .f8e8m0, .bf16 => |tag| switch (Ti) {
                    .comptime_int, .int => @unionInit(Value, @tagName(tag), @FieldType(Value, @tagName(tag)).fromF32(@floatFromInt(value))),
                    .comptime_float, .float => @unionInit(Value, @tagName(tag), @FieldType(Value, @tagName(tag)).fromF32(@floatCast(value))),
                    else => @panic("Could not create Value of type bf16 from value of type " ++ @typeName(T)),
                },
                inline .f16, .f32, .f64 => |tag| switch (Ti) {
                    .comptime_int, .int => @unionInit(Value, @tagName(tag), @floatFromInt(value)),
                    .comptime_float, .float => @unionInit(Value, @tagName(tag), @floatCast(value)),
                    else => @panic("Could not create Value of type " ++ @tagName(tag) ++ " from value of type " ++ @typeName(T)),
                },
                inline .i2, .i4, .i8, .i16, .i32, .i64, .u2, .u4, .u8, .u16, .u32, .u64 => |tag| switch (Ti) {
                    .comptime_int => blk: {
                        const OutT = @FieldType(Value, @tagName(tag));
                        if (value >= std.math.minInt(OutT) and value <= std.math.maxInt(OutT)) {
                            break :blk @unionInit(Value, @tagName(tag), @intCast(value));
                        } else {
                            @panic("Could not create Value of type " ++ @tagName(tag) ++ " from value of type " ++ @typeName(T));
                        }
                    },
                    .int => @unionInit(Value, @tagName(tag), @intCast(value)),
                    else => @panic("Could not create Value of type " ++ @tagName(tag) ++ " from value of type " ++ @typeName(T)),
                },
                .c64 => switch (T) {
                    C64 => .{ .c64 = value },
                    C128 => .{ .c64 = .{ .re = @floatCast(value.re), .im = @floatCast(value.im) } },
                    else => @panic("Could not create Value of type c64 from value of type " ++ @typeName(T)),
                },
                .c128 => switch (T) {
                    C64 => .{ .c128 = .{ .re = @floatCast(value.re), .im = @floatCast(value.im) } },
                    C128 => .{ .c128 = value },
                    else => @panic("Could not create Value of type c128 from value of type " ++ @typeName(T)),
                },
            };
        }

        test init {
            try std.testing.expectEqual(20.0, Value.init(.f16, 20).f16);
            try std.testing.expectEqual(20.5, Value.init(.f16, 20.5).f16);
            try std.testing.expectEqual(20, Value.init(.f16, @as(u8, 20)).f16);
            try std.testing.expectEqual(-20, Value.init(.f16, @as(i8, -20)).f16);
            try std.testing.expectEqual(2000.5, Value.init(.f16, @as(f32, 2000.5)).f16);

            try std.testing.expectEqual(true, Value.init(.bool, true).bool);

            try std.testing.expectEqual(10, Value.init(.u8, 10).u8);
            try std.testing.expectEqual(10, Value.init(.u8, @as(u16, 10)).u8);

            try std.testing.expectEqual(10, Value.init(.i8, 10).i8);
            try std.testing.expectEqual(10, Value.init(.i8, @as(u16, 10)).i8);
            try std.testing.expectEqual(-10, Value.init(.i8, -10).i8);
            try std.testing.expectEqual(-10, Value.init(.i8, @as(i16, -10)).i8);

            try std.testing.expectEqual(C64.init(1, 2), Value.init(.c64, C64.init(1, 2)).c64);
            try std.testing.expectEqual(C64.init(1, 2), Value.init(.c64, C128.init(1, 2)).c64);
            try std.testing.expectEqual(C128.init(1, 2), Value.init(.c128, C128.init(1, 2)).c128);
            try std.testing.expectEqual(C128.init(1, 2), Value.init(.c128, C64.init(1, 2)).c128);
        }

        pub fn dtype(self: Value) DataType {
            return std.meta.activeTag(self);
        }

        pub fn asBytes(data: *const Value) []const u8 {
            return switch (data.*) {
                inline else => |*value| std.mem.asBytes(value),
            };
        }

        pub fn as(self: Value, comptime T: type) T {
            // TODO allow more lossless conversions
            switch (@typeInfo(T)) {
                .bool => return self.bool,
                .float => switch (self) {
                    inline .f16, .f32, .f64 => |v| return @floatCast(v),
                    inline .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .bf16 => |v| return @floatCast(v.toF32()),
                    else => {},
                },
                .int => switch (self) {
                    inline .i4, .i8, .i16, .i32, .i64, .u4, .u8, .u16, .u32, .u64 => |v| return @intCast(v),
                    else => {},
                },
                else => {},
            }
            std.debug.panic("Unsupported conversion {} -> {s}", .{ self.dtype(), @typeName(T) });
        }
    };
};

pub fn mantissaSize(dtype: DataType) usize {
    return switch (dtype) {
        .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz => 3,
        .f8e5m2, .f8e5m2fnuz => 2,
        .f16 => 10,
        .bf16 => 7,
        .f32 => 23,
        .f64 => 52,
        else => @panic("Can't get mantissa size for a non-float dtype"),
    };
}

pub const Data = DataType.Value;
