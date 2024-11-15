pub inline fn divFloor(comptime T: type, numerator: anytype, denominator: anytype) T {
    return @divFloor(floatCast(T, numerator), floatCast(T, denominator));
}

pub inline fn divExact(comptime T: type, numerator: anytype, denominator: anytype) T {
    return @divExact(floatCast(T, numerator), floatCast(T, denominator));
}

pub inline fn divTrunc(comptime T: type, numerator: anytype, denominator: anytype) T {
    return @divTrunc(floatCast(T, numerator), floatCast(T, denominator));
}

pub inline fn floatCast(comptime T: type, x: anytype) T {
    return switch (@typeInfo(@TypeOf(x))) {
        .Float => @floatCast(x),
        else => @floatFromInt(x),
    };
}

pub inline fn intCast(comptime T: type, x: anytype) T {
    return switch (@typeInfo(@TypeOf(x))) {
        .Int => @intCast(x),
        else => @intFromFloat(x),
    };
}
