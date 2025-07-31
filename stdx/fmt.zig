const std = @import("std");

pub fn formatValue(value: anytype, full: std.fmt.Number, writer: anytype) !void {
    return switch (@typeInfo(@TypeOf(value))) {
        .comptime_float, .float => try formatFloatValue(value, full, writer),
        .comptime_int, .int => try formatIntValue(value, full, writer),
        else => try formatAnyValue(value, full, writer),
    };
}

pub fn formatFloatValue(value: anytype, full: std.fmt.Number, writer: *std.Io.Writer) !void {
    const x = switch (@typeInfo(@TypeOf(value))) {
        .@"struct" => value.toF32(),
        .float => value,
        else => @compileError("formatFloatValue expects a float, got: " ++ @typeName(@TypeOf(value))),
    };
    return writer.printFloat(x, full);
}

pub fn formatIntValue(value: anytype, full: std.fmt.Number, writer: *std.Io.Writer) !void {
    switch (@typeInfo(@TypeOf(value))) {
        .int => {},
        else => @compileError("formatIntValue expects an int, got: " ++ @typeName(@TypeOf(value))),
    }
    return writer.printInt(value, full.mode.base().?, full.case, .{ .alignment = full.alignment, .fill = full.fill });
}

pub fn formatAnyValue(value: anytype, full: std.fmt.Number, writer: *std.Io.Writer) !void {
    var buf: [48]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{any}", .{value}) catch blk: {
        buf[45..].* = "...".*;
        break :blk buf[0..];
    };
    return try writer.alignBufferOptions(s, .{ .alignment = full.alignment, .fill = full.fill });
}

pub fn formatSliceCustom(fmt_func: anytype, values: anytype, full: std.fmt.Number, writer: anytype) !void {
    // use the format "width" for the number of columns instead of individual width.
    const num_cols: usize = full.width orelse 12;
    var my_options = full;
    my_options.width = null;
    const n: usize = values.len;

    _ = try writer.write("{");
    if (n <= num_cols) {
        for (values, 0..) |v, i| {
            // Force inlining so that the switch and the buffer can be done once.
            try @call(.always_inline, fmt_func, .{ v, my_options, writer });
            if (i < n - 1) _ = try writer.write(",");
        }
    } else {
        const half = @divFloor(num_cols, 2);
        for (values[0..half]) |v| {
            try @call(.always_inline, fmt_func, .{ v, my_options, writer });
            _ = try writer.write(",");
        }
        _ = try writer.write(" ..., ");
        for (values[n - half ..], 0..) |v, i| {
            try @call(.always_inline, fmt_func, .{ v, my_options, writer });
            if (i < half - 1) _ = try writer.write(",");
        }
    }
    _ = try writer.write("}");
}

pub fn formatAny(values: anytype, full: std.fmt.Number, writer: anytype) !void {
    return try formatSliceCustom(formatAnyValue, values, full, writer);
}

pub fn formatFloatSlice(values: anytype, full: std.fmt.Number, writer: anytype) !void {
    return try formatSliceCustom(formatFloatValue, values, full, writer);
}

pub fn formatIntSlice(values: anytype, full: std.fmt.Number, writer: anytype) !void {
    return try formatSliceCustom(formatIntValue, values, full, writer);
}

pub fn formatAnySlice(values: anytype, full: std.fmt.Number, writer: anytype) !void {
    return try formatSliceCustom(formatAnyValue, values, full, writer);
}
