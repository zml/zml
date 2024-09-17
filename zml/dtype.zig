const std = @import("std");
const floats = @import("floats.zig");

const C64 = std.math.Complex(f32);
const C128 = std.math.Complex(f64);

pub const DataType = enum(u8) {
    bool,
    f8e4m3b11fnuz,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,
    bf16,
    f16,
    f32,
    f64,
    i4,
    i8,
    i16,
    i32,
    i64,
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
            .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .bf16, .f16, .f32, .f64 => .float,
            .i4, .i8, .i16, .i32, .i64, .u4, .u8, .u16, .u32, .u64 => .integer,
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
            floats.Float8E4M3B11FNUZ => .f8e4m3b11fnuz,
            floats.Float8E4M3FN => .f8e4m3fn,
            floats.Float8E4M3FNUZ => .f8e4m3fnuz,
            floats.Float8E5M2 => .f8e5m2,
            floats.Float8E5M2FNUZ => .f8e5m2fnuz,
            floats.BFloat16 => .bf16,
            f16 => .f16,
            f32 => .f32,
            f64 => .f64,
            bool => .bool,
            i4 => .i4,
            i8 => .i8,
            i16 => .i16,
            i32 => .i32,
            i64 => .i64,
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
        if (type_info != .Pointer) {
            @compileError("`initFromSlice` expects a slice, got " ++ @tagName(type_info));
        }

        return switch (type_info.Pointer.size) {
            .Slice, .C, .Many => DataType.fromZigType(type_info.Pointer.child),
            .One => b: {
                const child_type_info = @typeInfo(type_info.Pointer.child);
                break :b DataType.fromZigType(child_type_info.Array.child);
            },
        };
    }

    pub fn toZigType(comptime dtype: DataType) type {
        return switch (dtype) {
            inline else => |tag| std.meta.TagPayload(Data, tag),
        };
    }

    pub fn isSignedInt(dtype: DataType) bool {
        return switch (dtype) {
            .i4, .i8, .i16, .i32, .i64 => true,
            else => false,
        };
    }

    pub fn sizeOf(self: DataType) u16 {
        return switch (self) {
            inline else => |tag| @sizeOf(std.meta.TagPayload(Data, tag)),
        };
    }

    pub fn bitSizeOf(self: DataType) u16 {
        return switch (self) {
            inline else => |tag| @bitSizeOf(std.meta.TagPayload(Data, tag)),
        };
    }

    pub fn alignOf(self: DataType) u29 {
        return switch (self) {
            inline else => |tag| @alignOf(std.meta.TagPayload(Data, tag)),
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

    pub fn zero(dtype: DataType) Data {
        return Data.init(dtype, 0);
    }

    pub fn one(dtype: DataType) Data {
        return Data.init(dtype, 1);
    }

    pub fn minValue(dtype: DataType) Data {
        return switch (dtype) {
            .bool => .{ .bool = false },
            inline .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2fnuz => |tag| @unionInit(Data, @tagName(tag), std.meta.FieldType(Data, tag).zero()),
            inline .f8e5m2, .bf16 => |tag| @unionInit(Data, @tagName(tag), std.meta.FieldType(Data, tag).minusInf()),
            inline .f16, .f32, .f64 => |tag| @unionInit(Data, @tagName(tag), -std.math.inf(std.meta.FieldType(Data, tag))),
            inline .i4, .i8, .i16, .i32, .i64, .u4, .u8, .u16, .u32, .u64 => |tag| @unionInit(Data, @tagName(tag), std.math.minInt(std.meta.FieldType(Data, tag))),
            inline else => |tag| @panic("Unsupported type: " ++ @tagName(tag)),
        };
    }

    pub fn maxValue(dtype: DataType) Data {
        return switch (dtype) {
            .bool => .{ .bool = true },
            inline .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2fnuz => |tag| @panic("DataType doesn't have a max value: " ++ @tagName(tag)),
            inline .f8e5m2, .bf16 => |tag| @unionInit(Data, @tagName(tag), std.meta.FieldType(Data, tag).inf()),
            inline .f16, .f32, .f64 => |tag| @unionInit(Data, @tagName(tag), std.math.inf(std.meta.FieldType(Data, tag))),
            inline .i4, .i8, .i16, .i32, .i64, .u4, .u8, .u16, .u32, .u64 => |tag| @unionInit(Data, @tagName(tag), std.math.maxInt(std.meta.FieldType(Data, tag))),
            inline .c64, .c128 => |tag| @panic("DataType doesn't have a max value: " ++ @tagName(tag)),
        };
    }

    pub fn constant(dtype: DataType, value: anytype) Data {
        return Data.init(dtype, value);
    }
};

pub const Data = union(DataType) {
    bool: bool,
    f8e4m3b11fnuz: floats.Float8E4M3B11FNUZ,
    f8e4m3fn: floats.Float8E4M3FN,
    f8e4m3fnuz: floats.Float8E4M3FNUZ,
    f8e5m2: floats.Float8E5M2,
    f8e5m2fnuz: floats.Float8E5M2FNUZ,
    bf16: floats.BFloat16,
    f16: f16,
    f32: f32,
    f64: f64,
    i4: i4,
    i8: i8,
    i16: i16,
    i32: i32,
    i64: i64,
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
    pub fn init(dtype: DataType, value: anytype) Data {
        const T = @TypeOf(value);
        const Ti = @typeInfo(T);

        return switch (dtype) {
            .bool => switch (Ti) {
                .Bool => .{ .bool = value },
                .ComptimeInt, .Int, .ComptimeFloat, .Float => .{ .bool = value != 0 },
                else => @panic("Could not create Data of type bool from value of type " ++ @typeName(T)),
            },
            inline .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .bf16 => |tag| switch (Ti) {
                .ComptimeInt, .Int => @unionInit(Data, @tagName(tag), std.meta.FieldType(Data, tag).fromF32(@floatFromInt(value))),
                .ComptimeFloat, .Float => @unionInit(Data, @tagName(tag), std.meta.FieldType(Data, tag).fromF32(@floatCast(value))),
                else => @panic("Could not create Data of type bf16 from value of type " ++ @typeName(T)),
            },
            inline .f16, .f32, .f64 => |tag| switch (Ti) {
                .ComptimeInt, .Int => @unionInit(Data, @tagName(tag), @floatFromInt(value)),
                .ComptimeFloat, .Float => @unionInit(Data, @tagName(tag), @floatCast(value)),
                else => @panic("Could not create Data of type " ++ @tagName(tag) ++ " from value of type " ++ @typeName(T)),
            },
            inline .i4, .i8, .i16, .i32, .i64, .u4, .u8, .u16, .u32, .u64 => |tag| switch (Ti) {
                .ComptimeInt => blk: {
                    const OutT = std.meta.FieldType(Data, tag);
                    if (value >= std.math.minInt(OutT) and value <= std.math.maxInt(OutT)) {
                        break :blk @unionInit(Data, @tagName(tag), @intCast(value));
                    } else {
                        @panic("Could not create Data of type " ++ @tagName(tag) ++ " from value of type " ++ @typeName(T));
                    }
                },
                .Int => @unionInit(Data, @tagName(tag), @intCast(value)),
                else => @panic("Could not create Data of type " ++ @tagName(tag) ++ " from value of type " ++ @typeName(T)),
            },
            .c64 => switch (T) {
                C64 => .{ .c64 = value },
                C128 => .{ .c64 = .{ .re = @floatCast(value.re), .im = @floatCast(value.im) } },
                else => @panic("Could not create Data of type c64 from value of type " ++ @typeName(T)),
            },
            .c128 => switch (T) {
                C64 => .{ .c128 = .{ .re = @floatCast(value.re), .im = @floatCast(value.im) } },
                C128 => .{ .c128 = value },
                else => @panic("Could not create Data of type c128 from value of type " ++ @typeName(T)),
            },
        };
    }

    test init {
        try std.testing.expectEqual(20.0, Data.init(.f16, 20).f16);
        try std.testing.expectEqual(20.5, Data.init(.f16, 20.5).f16);
        try std.testing.expectEqual(20, Data.init(.f16, @as(u8, 20)).f16);
        try std.testing.expectEqual(-20, Data.init(.f16, @as(i8, -20)).f16);
        try std.testing.expectEqual(2000.5, Data.init(.f16, @as(f32, 2000.5)).f16);

        try std.testing.expectEqual(true, Data.init(.bool, true).bool);

        try std.testing.expectEqual(10, Data.init(.u8, 10).u8);
        try std.testing.expectEqual(10, Data.init(.u8, @as(u16, 10)).u8);

        try std.testing.expectEqual(10, Data.init(.i8, 10).i8);
        try std.testing.expectEqual(10, Data.init(.i8, @as(u16, 10)).i8);
        try std.testing.expectEqual(-10, Data.init(.i8, -10).i8);
        try std.testing.expectEqual(-10, Data.init(.i8, @as(i16, -10)).i8);

        try std.testing.expectEqual(C64.init(1, 2), Data.init(.c64, C64.init(1, 2)).c64);
        try std.testing.expectEqual(C64.init(1, 2), Data.init(.c64, C128.init(1, 2)).c64);
        try std.testing.expectEqual(C128.init(1, 2), Data.init(.c128, C128.init(1, 2)).c128);
        try std.testing.expectEqual(C128.init(1, 2), Data.init(.c128, C64.init(1, 2)).c128);
    }

    pub fn dataType(self: Data) DataType {
        return std.meta.activeTag(self);
    }

    pub fn constSlice(data: *const Data) []const u8 {
        return switch (data.*) {
            inline else => |*value| std.mem.asBytes(value),
        };
    }

    pub fn as(self: Data, comptime T: type) T {
        // TODO allow more lossless conversions
        switch (@typeInfo(T)) {
            .Bool => return self.bool,
            .Float => switch (self) {
                inline .f16, .f32, .f64 => |v| return @floatCast(v),
                inline .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .bf16 => |v| return @floatCast(v.toF32()),
                else => {},
            },
            .Int => switch (self) {
                inline .i4, .i8, .i16, .i32, .i64, .u4, .u8, .u16, .u32, .u64 => |v| return @intCast(v),
                else => {},
            },
            else => {},
        }
        std.debug.panic("Unsupported conversion {} -> {s}", .{ self.dataType(), @typeName(T) });
    }
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
