pub inline fn divFloat(comptime T: type, numerator: anytype, denominator: anytype) T {
    return floatCast(T, numerator) / floatCast(T, denominator);
}

pub inline fn floatCast(comptime T: type, x: anytype) T {
    return switch (@typeInfo(@TypeOf(x))) {
        .float => @floatCast(x),
        else => @floatFromInt(x),
    };
}

pub inline fn intCast(comptime T: type, x: anytype) T {
    return switch (@typeInfo(@TypeOf(x))) {
        .int => @intCast(x),
        else => @intFromFloat(x),
    };
}

pub inline fn roundeven(x: anytype) @TypeOf(x) {
    return struct {
        extern fn @"llvm.roundeven"(@TypeOf(x)) @TypeOf(x);
    }.@"llvm.roundeven"(x);
}
