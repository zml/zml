const std = @import("std");

const mlir = @import("mlir");

/// An MSL scalar type.
pub const Scalar = enum {
    bool,
    char,
    uchar,
    short,
    ushort,
    int,
    uint,
    long,
    ulong,
    half,
    bfloat,
    float,

    pub fn isFloat(self: Scalar) bool {
        return switch (self) {
            .half, .bfloat, .float => true,
            else => false,
        };
    }

    pub fn isUnsigned(self: Scalar) bool {
        return switch (self) {
            .bool, .uchar, .ushort, .uint, .ulong => true,
            else => false,
        };
    }

    pub fn bits(self: Scalar) u16 {
        return switch (self) {
            .bool => 1,
            .char, .uchar => 8,
            .short, .ushort, .half, .bfloat => 16,
            .int, .uint, .float => 32,
            .long, .ulong => 64,
        };
    }

    /// The size in memory: a bool takes a byte.
    pub fn size(self: Scalar) u64 {
        return if (self == .bool) 1 else self.bits() / 8;
    }

    /// The C++ integer conversion rank.
    fn rank(self: Scalar) u8 {
        return switch (self) {
            .bool => 0,
            .char, .uchar => 1,
            .short, .ushort => 2,
            .int, .uint => 3,
            .long, .ulong => 4,
            else => unreachable,
        };
    }

    fn unsignedOf(self: Scalar) Scalar {
        return switch (self) {
            .char => .uchar,
            .short => .ushort,
            .int => .uint,
            .long => .ulong,
            else => self,
        };
    }

    /// C++ integral promotion: types narrower than int become int.
    pub fn promoted(self: Scalar) Scalar {
        return switch (self) {
            .bool, .char, .uchar, .short, .ushort => .int,
            else => self,
        };
    }

    pub fn mlirType(self: Scalar, ctx: *mlir.Context) *const mlir.Type {
        return switch (self) {
            .bool => .int(ctx, .i1),
            .char, .uchar => .int(ctx, .i8),
            .short, .ushort => .int(ctx, .i16),
            .int, .uint => .int(ctx, .i32),
            .long, .ulong => .int(ctx, .i64),
            .half => .float(ctx, .f16),
            .bfloat => .float(ctx, .bf16),
            .float => .float(ctx, .f32),
        };
    }
};

/// The C++ usual arithmetic conversions of two scalar operand types.
pub fn commonScalar(a: Scalar, b: Scalar) Scalar {
    if (a.isFloat() or b.isFloat()) {
        if (!b.isFloat()) return a;
        if (!a.isFloat()) return b;
        if (a == b) return a;
        if (a == .float or b == .float) return .float;
        std.debug.panic("dsl/metal: no common type of {t} and {t}", .{ a, b });
    }
    const x = a.promoted();
    const y = b.promoted();
    if (x == y) return x;
    if (x.isUnsigned() == y.isUnsigned()) return if (x.rank() >= y.rank()) x else y;
    const u = if (x.isUnsigned()) x else y;
    const s = if (x.isUnsigned()) y else x;
    if (u.rank() >= s.rank()) return u;
    if (s.bits() > u.bits()) return s;
    return s.unsignedOf();
}

/// An MSL scalar or vector type: `float`, `uint3`, `bfloat4`.
pub const Type = struct {
    scalar: Scalar,
    /// 1 for a scalar, else the vector length (2, 3 or 4).
    len: u8 = 1,

    pub const @"bool": Type = .{ .scalar = .bool };
    pub const char: Type = .{ .scalar = .char };
    pub const uchar: Type = .{ .scalar = .uchar };
    pub const short: Type = .{ .scalar = .short };
    pub const ushort: Type = .{ .scalar = .ushort };
    pub const int: Type = .{ .scalar = .int };
    pub const uint: Type = .{ .scalar = .uint };
    pub const long: Type = .{ .scalar = .long };
    pub const ulong: Type = .{ .scalar = .ulong };
    pub const half: Type = .{ .scalar = .half };
    pub const bfloat: Type = .{ .scalar = .bfloat };
    pub const float: Type = .{ .scalar = .float };
    pub const bool2: Type = .vec(.bool, 2);
    pub const bool3: Type = .vec(.bool, 3);
    pub const bool4: Type = .vec(.bool, 4);
    pub const char2: Type = .vec(.char, 2);
    pub const char3: Type = .vec(.char, 3);
    pub const char4: Type = .vec(.char, 4);
    pub const uchar2: Type = .vec(.uchar, 2);
    pub const uchar3: Type = .vec(.uchar, 3);
    pub const uchar4: Type = .vec(.uchar, 4);
    pub const short2: Type = .vec(.short, 2);
    pub const short3: Type = .vec(.short, 3);
    pub const short4: Type = .vec(.short, 4);
    pub const ushort2: Type = .vec(.ushort, 2);
    pub const ushort3: Type = .vec(.ushort, 3);
    pub const ushort4: Type = .vec(.ushort, 4);
    pub const int2: Type = .vec(.int, 2);
    pub const int3: Type = .vec(.int, 3);
    pub const int4: Type = .vec(.int, 4);
    pub const uint2: Type = .vec(.uint, 2);
    pub const uint3: Type = .vec(.uint, 3);
    pub const uint4: Type = .vec(.uint, 4);
    pub const long2: Type = .vec(.long, 2);
    pub const long3: Type = .vec(.long, 3);
    pub const long4: Type = .vec(.long, 4);
    pub const ulong2: Type = .vec(.ulong, 2);
    pub const ulong3: Type = .vec(.ulong, 3);
    pub const ulong4: Type = .vec(.ulong, 4);
    pub const half2: Type = .vec(.half, 2);
    pub const half3: Type = .vec(.half, 3);
    pub const half4: Type = .vec(.half, 4);
    pub const bfloat2: Type = .vec(.bfloat, 2);
    pub const bfloat3: Type = .vec(.bfloat, 3);
    pub const bfloat4: Type = .vec(.bfloat, 4);
    pub const float2: Type = .vec(.float, 2);
    pub const float3: Type = .vec(.float, 3);
    pub const float4: Type = .vec(.float, 4);

    /// `vec<scalar, len>`.
    pub fn vec(scalar: Scalar, len: u8) Type {
        std.debug.assert(len >= 2 and len <= 4);
        return .{ .scalar = scalar, .len = len };
    }

    pub fn isVector(self: Type) bool {
        return self.len > 1;
    }

    pub fn element(self: Type) Type {
        return .{ .scalar = self.scalar };
    }

    /// The same shape with another element type.
    pub fn withScalar(self: Type, scalar: Scalar) Type {
        return .{ .scalar = scalar, .len = self.len };
    }

    /// The type in memory, where a bool is a byte, as clang stores it.
    pub fn memory(self: Type) Type {
        return if (self.scalar == .bool) self.withScalar(.uchar) else self;
    }

    /// `sizeof`: a 3-element vector takes the space of 4.
    pub fn size(self: Type) u64 {
        const len: u64 = if (self.len == 3) 4 else self.len;
        return len * self.scalar.size();
    }

    /// `alignof`, which is the size for MSL scalars and vectors.
    pub fn alignment(self: Type) u64 {
        return self.size();
    }

    pub fn eql(self: Type, other: Type) bool {
        return self.scalar == other.scalar and self.len == other.len;
    }

    /// The MSL spelling, `uint3`.
    pub fn name(self: Type) []const u8 {
        const names = comptime blk: {
            const scalars = @typeInfo(Scalar).@"enum".field_names;
            var table: [scalars.len][4][]const u8 = undefined;
            for (scalars, 0..) |scalar, i| {
                table[i] = .{ scalar, scalar ++ "2", scalar ++ "3", scalar ++ "4" };
            }
            break :blk table;
        };
        return names[@backingInt(self.scalar)][self.len - 1];
    }

    pub fn mlirType(self: Type, ctx: *mlir.Context) *const mlir.Type {
        const scalar = self.scalar.mlirType(ctx);
        return if (self.isVector()) .vector(&.{self.len}, scalar) else scalar;
    }
};

/// A field of a constant buffer struct: `float s[3]` is
/// `.{ .name = "s", .type = .float, .count = 3 }`.
pub const StructField = struct {
    name: []const u8,
    type: Type,
    /// The array length, zero for a non-array field.
    count: u32 = 0,
};

/// A struct passed by constant reference, laid out as MSL lays it out.
pub const Struct = struct {
    name: []const u8,
    fields: []const StructField,

    pub fn offset(self: Struct, index: usize) u64 {
        var end: u64 = 0;
        for (self.fields[0..index]) |field| {
            end = std.mem.alignForward(u64, end, field.type.alignment());
            end += field.type.size() * @max(field.count, 1);
        }
        return std.mem.alignForward(u64, end, self.fields[index].type.alignment());
    }

    pub fn alignment(self: Struct) u64 {
        var a: u64 = 1;
        for (self.fields) |field| a = @max(a, field.type.alignment());
        return a;
    }

    pub fn size(self: Struct) u64 {
        const last = self.fields.len - 1;
        const end = self.offset(last) + self.fields[last].type.size() * @max(self.fields[last].count, 1);
        return std.mem.alignForward(u64, end, self.alignment());
    }

    pub fn fieldIndex(self: Struct, field_name: []const u8) usize {
        for (self.fields, 0..) |field, i| if (std.mem.eql(u8, field.name, field_name)) return i;
        std.debug.panic("dsl/metal: struct {s} has no field {s}", .{ self.name, field_name });
    }
};

test commonScalar {
    try std.testing.expectEqual(Scalar.uint, commonScalar(.uint, .int));
    try std.testing.expectEqual(Scalar.int, commonScalar(.ushort, .short));
    try std.testing.expectEqual(Scalar.long, commonScalar(.long, .uint));
    try std.testing.expectEqual(Scalar.ulong, commonScalar(.ulong, .int));
    try std.testing.expectEqual(Scalar.float, commonScalar(.int, .float));
    try std.testing.expectEqual(Scalar.half, commonScalar(.half, .uint));
    try std.testing.expectEqual(Scalar.float, commonScalar(.half, .float));
}

test Struct {
    const args: Struct = .{ .name = "A", .fields = &.{
        .{ .name = "n", .type = .int },
        .{ .name = "s", .type = .float, .count = 3 },
        .{ .name = "v", .type = .uint2 },
        .{ .name = "t", .type = .short },
    } };
    try std.testing.expectEqual(0, args.offset(0));
    try std.testing.expectEqual(4, args.offset(1));
    try std.testing.expectEqual(16, args.offset(2));
    try std.testing.expectEqual(24, args.offset(3));
    try std.testing.expectEqual(32, args.size());
    try std.testing.expectEqualStrings("bfloat4", Type.bfloat4.name());
    try std.testing.expectEqual(16, Type.float3.size());
}
