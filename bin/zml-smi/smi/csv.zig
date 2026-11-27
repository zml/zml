const std = @import("std");
const smi = @import("smi.zig");

pub fn write(writer: *std.Io.Writer, collector: *smi.Collector) !void {
    inline for (comptime std.meta.tags(smi.Target)) |target| {
        try writeTable(writer, collector, target);
    }
}

fn writeTable(writer: *std.Io.Writer, collector: *smi.Collector, comptime target: smi.Target) !void {
    const info_fields = @typeInfo(@FieldType(smi.Device.Info, @tagName(target))).@"struct".field_names;
    const fields = @typeInfo(@FieldType(smi.Sample, @tagName(target))).@"struct".field_names;

    var header_written = false;
    for (collector.devices.items, 0..) |device, i| {
        if (device.info != target) continue;
        if (!header_written) {
            try writer.writeAll("index,type,remote,name");
            inline for (info_fields) |name| try writer.writeAll("," ++ name);
            inline for (fields) |field| try writer.writeAll("," ++ field);
            try writer.writeAll("\n");
            header_written = true;
        }
        try writer.print("{d},{t},{any}", .{ i, target, device.remote });
        try writeField(writer, device.name orelse "");
        const info = @field(device.info, @tagName(target));
        inline for (info_fields) |name| try writeField(writer, @field(info, name) orelse "");

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
