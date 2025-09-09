const std = @import("std");

/// Properly format a slice of numbers.
pub fn slice(any_slice: anytype) FmtSlice(std.meta.Elem(@TypeOf(any_slice))) {
    return .{ .slice = any_slice };
}

fn FmtSlice(T: type) type {
    return struct {
        slice: []const T,

        pub fn formatNumber(f: @This(), writer: *std.io.Writer, n: std.fmt.Number) std.io.Writer.Error!void {
            return switch (@typeInfo(T)) {
                .comptime_float, .float => try formatFloatSlice(f.slice, n, writer),
                .comptime_int, .int => try formatIntSlice(f.slice, n, writer),
                .bool => try formatBoolSlice(f.slice, n, writer),
                .@"struct" => if (@hasField(T, "re") and @hasField(T, "im")) {
                    try formatComplexSlice(f.slice, n, writer);
                } else if (@hasDecl(T, "toF32")) {
                    try formatFloatSlice(f.slice, n, writer);
                } else {
                    try formatSliceAny(f.slice, n, writer);
                },
                else => @compileError("FmtSlice doesn't support type: " ++ @typeName(T)),
            };
        }
    };
}

pub fn formatFloat(value: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    const x = switch (@typeInfo(@TypeOf(value))) {
        .@"struct" => value.toF32(),
        .float => value,
        else => @compileError("formatFloat expects a float, got: " ++ @typeName(@TypeOf(value))),
    };
    return writer.printFloat(x, spec);
}

pub fn formatInt(value: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    switch (@typeInfo(@TypeOf(value))) {
        .int => {},
        else => @compileError("formatInt expects an int, got: " ++ @typeName(@TypeOf(value))),
    }
    return writer.printInt(value, spec.mode.base().?, spec.case, .{ .alignment = spec.alignment, .fill = spec.fill });
}

pub fn formatComplex(value: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    try writer.writeAll(".{.re=");
    try writer.printFloat(value.re, spec);
    try writer.writeAll(", .im=");
    try writer.printFloat(value.im, spec);
    try writer.writeAll("}");
}

pub fn formatBool(value: bool, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    try writer.alignBufferOptions(if (value) "1" else "0", .{ .alignment = spec.alignment, .fill = spec.fill });
}

pub fn formatAny(value: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    var buf: [48]u8 = undefined;
    const T = @TypeOf(value);
    const fmt = if (@hasDecl(T, "formatNumber")) "{d}" else "{f}";

    const s = std.fmt.bufPrint(&buf, fmt, .{value}) catch blk: {
        buf[45..].* = "...".*;
        break :blk buf[0..];
    };
    return try writer.alignBufferOptions(s, .{ .alignment = spec.alignment, .fill = spec.fill });
}

pub fn formatSliceCustom(fmt_func: anytype, values: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    // use the format "width" for the number of columns instead of individual width.
    const num_cols: usize = spec.width orelse 12;
    var my_options = spec;
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

pub fn formatSliceAny(values: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    return try formatSliceCustom(formatAny, values, spec, writer);
}

pub fn formatFloatSlice(values: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    return try formatSliceCustom(formatFloat, values, spec, writer);
}

pub fn formatIntSlice(values: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    return try formatSliceCustom(formatInt, values, spec, writer);
}

pub fn formatComplexSlice(values: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    return try formatSliceCustom(formatComplex, values, spec, writer);
}

pub fn formatBoolSlice(values: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    return try formatSliceCustom(formatBool, values, spec, writer);
}

/// Format a struct using `format` method of subfields when possible.
pub fn any(any_val: anytype) FmtAny(@TypeOf(any_val)) {
    return .{ .data = any_val };
}

fn FmtAny(Data: type) type {
    return struct {
        data: Data,

        pub inline fn format(self: @This(), writer: *std.io.Writer) std.io.Writer.Error!void {
            try printValue(writer, .{}, self.data, std.options.fmt_max_depth);
        }
    };
}

/// Fix up of std.io.Writer.printValue that uses `format` method of subfields when possible
fn printValue(
    w: *std.io.Writer,
    options: std.fmt.Options,
    value: anytype,
    max_depth: usize,
) std.io.Writer.Error!void {
    const T = @TypeOf(value);
    if (std.meta.hasMethod(T, "format")) {
        return try value.format(w);
    }
    if (std.meta.hasMethod(T, "formatNumber")) {
        return try value.formatNumber(w, options.toNumber(.decimal, .lower));
    }

    if (max_depth == 0) {
        try w.writeAll(".{ ... }");
        return;
    }

    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            try w.writeAll(".{ ");
            inline for (info.fields, 0..) |f, i| {
                if (i > 0) try w.writeAll(", ");

                if (!info.is_tuple) {
                    try w.writeByte('.');
                    try w.writeAll(f.name);
                    try w.writeAll(" = ");
                }
                try printValue(w, options, @field(value, f.name), max_depth - 1);
            }
            try w.writeAll(" }");
        },
        inline else => try w.printValue("any", options, value, max_depth - 1),
    }
}
