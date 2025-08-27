const std = @import("std");

pub const Fmt = union(enum) {
    int: IntFmt,
    float: FloatFmt,
    generic: void,

    pub fn parse(T: type, comptime fmt_: []const u8) Fmt {
        return switch (@typeInfo(T)) {
            .float, .comptime_float => .{ .float = FloatFmt.parseComptime(fmt_) },
            .int, .comptime_int => .{ .int = IntFmt.parseComptime(fmt_) },
            else => .{ .generic = {} },
        };
    }
};

pub const FullFormatOptions = struct {
    fmt: Fmt,
    options: std.fmt.FormatOptions,
};

pub const IntFmt = struct {
    base: u8,
    case: std.fmt.Case = .lower,

    pub fn parseComptime(comptime fmt_: []const u8) IntFmt {
        return parse(fmt_) catch @panic("invalid fmt for int: " ++ fmt_);
    }

    pub fn parse(fmt_: []const u8) error{InvalidArgument}!IntFmt {
        return if (fmt_.len == 0 or std.mem.eql(u8, fmt_, "d"))
            .{ .base = 10, .case = .lower }
        else if (std.mem.eql(u8, fmt_, "x"))
            .{ .base = 16, .case = .lower }
        else if (std.mem.eql(u8, fmt_, "X"))
            .{ .base = 16, .case = .upper }
        else if (std.mem.eql(u8, fmt_, "o"))
            .{ .base = 8, .case = .upper }
        else
            // TODO: unicode/ascii
            error.InvalidArgument;
    }
};

pub const FloatFmt = enum(u8) {
    scientific = @intFromEnum(std.fmt.Number.Mode.scientific),
    decimal = @intFromEnum(std.fmt.Number.Mode.decimal),
    hex,

    pub fn parseComptime(comptime fmt_: []const u8) FloatFmt {
        return parse(fmt_) catch @panic("invalid fmt for float: " ++ fmt_);
    }

    pub fn parse(fmt_: []const u8) error{InvalidArgument}!FloatFmt {
        return if (fmt_.len == 0 or std.mem.eql(u8, fmt_, "e"))
            .scientific
        else if (std.mem.eql(u8, fmt_, "d"))
            .decimal
        else if (std.mem.eql(u8, fmt_, "x"))
            .hex
        else
            error.InvalidArgument;
    }
};

pub fn formatValue(value: anytype, full: FullFormatOptions, writer: anytype) !void {
    return switch (@typeInfo(@TypeOf(value))) {
        .comptime_float, .float => try formatFloatValue(value, full, writer),
        .comptime_int, .int => try formatIntValue(value, full, writer),
        else => try formatAnyValue(value, full, writer),
    };
}

pub fn formatFloatValue(value: anytype, full: FullFormatOptions, writer: *std.Io.Writer) !void {
    const x = switch (@typeInfo(@TypeOf(value))) {
        .@"struct" => value.toF32(),
        .float => value,
        else => @compileError("formatFloatValue expects a float, got: " ++ @typeName(@TypeOf(value))),
    };
    try switch (full.fmt.float) {
        .scientific => writer.printFloat(x, .{ .mode = .scientific, .precision = full.options.precision }),
        .decimal => writer.printFloat(x, .{ .mode = .decimal, .precision = full.options.precision }),
        .hex => writer.printFloatHexOptions(x, .{ .mode = .hex }),
    };
}

pub fn formatIntValue(value: anytype, full: FullFormatOptions, writer: *std.Io.Writer) !void {
    switch (@typeInfo(@TypeOf(value))) {
        .int => {},
        else => @compileError("formatIntValue expects an int, got: " ++ @typeName(@TypeOf(value))),
    }
    return writer.printInt(value, full.fmt.int.base, full.fmt.int.case, full.options);
}

pub fn formatAnyValue(value: anytype, full: FullFormatOptions, writer: *std.Io.Writer) !void {
    var buf: [48]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{any}", .{value}) catch blk: {
        buf[45..].* = "...".*;
        break :blk buf[0..];
    };
    return try writer.alignBufferOptions(s, full.options);
}

pub fn formatSliceCustom(fmt_func: anytype, values: anytype, full: FullFormatOptions, writer: anytype) !void {

    // Write first rows
    const num_cols: usize = full.options.width orelse 12;
    const n: usize = values.len;
    _ = try writer.write("{");
    if (n <= num_cols) {
        for (values, 0..) |v, i| {
            // Force inlining so that the switch and the buffer can be done once.
            try @call(.always_inline, fmt_func, .{ v, full, writer });
            if (i < n - 1) _ = try writer.write(",");
        }
    } else {
        const half = @divFloor(num_cols, 2);
        for (values[0..half]) |v| {
            try @call(.always_inline, fmt_func, .{ v, full, writer });
            _ = try writer.write(",");
        }
        _ = try writer.write(" ..., ");
        for (values[n - half ..], 0..) |v, i| {
            try @call(.always_inline, fmt_func, .{ v, full, writer });
            if (i < half - 1) _ = try writer.write(",");
        }
    }
    _ = try writer.write("}");
}

pub fn formatAny(values: anytype, full: FullFormatOptions, writer: anytype) !void {
    return try formatSliceCustom(formatAnyValue, values, full, writer);
}

pub fn formatFloatSlice(values: anytype, full: FullFormatOptions, writer: anytype) !void {
    return try formatSliceCustom(formatFloatValue, values, full, writer);
}

pub fn formatIntSlice(values: anytype, full: FullFormatOptions, writer: anytype) !void {
    return try formatSliceCustom(formatIntValue, values, full, writer);
}

pub fn formatAnySlice(values: anytype, full: FullFormatOptions, writer: anytype) !void {
    return try formatSliceCustom(formatAnyValue, values, full, writer);
}
