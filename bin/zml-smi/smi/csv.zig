const std = @import("std");
const smi = @import("smi.zig");

pub fn write(writer: *std.Io.Writer, collector: *smi.Collector) !void {
    inline for (@typeInfo(smi.Target).@"enum".field_names) |name| {
        try writeTable(writer, collector, @field(smi.Target, name));
    }
}

fn writeTable(writer: *std.Io.Writer, collector: *smi.Collector, comptime target: smi.Target) !void {
    const fields = @typeInfo(@FieldType(smi.Sample, @tagName(target))).@"struct".field_names;

    var rows: usize = 0;
    var names = false;
    var drivers = false;
    var key_buffer: [16][]const u8 = undefined;
    var keys: std.ArrayList([]const u8) = .initBuffer(&key_buffer);
    for (collector.devices.items) |device| {
        if (device.target != target) continue;
        rows += 1;
        names = names or device.name != null;
        drivers = drivers or device.driver_version != null;
        for (device.details) |detail| {
            if (!contains(keys.items, detail.key)) keys.appendBounded(detail.key) catch {};
        }
    }
    if (rows == 0) return;

    try writer.writeAll("index,type,remote");
    if (names) try writer.writeAll(",name");
    if (drivers) try writer.writeAll(",driver_version");
    for (keys.items) |key| try writeField(writer, key);
    inline for (fields) |field| try writer.writeAll("," ++ field);
    try writer.writeAll("\n");

    for (collector.devices.items, 0..) |device, i| {
        if (device.target != target) continue;
        try writer.print("{d},{t},{any}", .{ i, target, device.remote });
        if (names) try writeField(writer, device.name orelse "");
        if (drivers) try writeField(writer, device.driver_version orelse "");
        for (keys.items) |key| try writeField(writer, detailValue(device, key) orelse "");

        const sample = @field(collector.sample(@intCast(i)), @tagName(target));
        inline for (fields) |field| {
            try writer.writeAll(",");
            if (@field(sample, field)) |value| switch (@typeInfo(@TypeOf(value))) {
                .@"enum" => try writer.writeAll(@tagName(value)),
                else => try writer.print("{d}", .{value}),
            };
        }
        try writer.writeAll("\n");
    }
}

fn contains(keys: []const []const u8, key: []const u8) bool {
    for (keys) |other| {
        if (std.mem.eql(u8, other, key)) return true;
    }
    return false;
}

fn detailValue(device: smi.Device, key: []const u8) ?[]const u8 {
    for (device.details) |item| {
        if (std.mem.eql(u8, item.key, key)) return item.value;
    }
    return null;
}

fn writeField(writer: *std.Io.Writer, text: []const u8) !void {
    try writer.writeAll(",");
    if (std.mem.findAny(u8, text, ",\"\r\n") == null) return writer.writeAll(text);

    try writer.writeAll("\"");
    var rest = text;
    while (std.mem.cutScalar(u8, rest, '"')) |parts| {
        const before, const after = parts;
        try writer.writeAll(before);
        try writer.writeAll("\"\"");
        rest = after;
    }
    try writer.writeAll(rest);
    try writer.writeAll("\"");
}

test writeField {
    var buf: [128]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);

    try writeField(&w, "plain");
    try writeField(&w, "2026.4.0, 32ffd4e");
    try writeField(&w, "a\"b\"c");
    try writeField(&w, "line1\nline2");

    try std.testing.expectEqualStrings(",plain,\"2026.4.0, 32ffd4e\",\"a\"\"b\"\"c\",\"line1\nline2\"", w.buffered());
}
