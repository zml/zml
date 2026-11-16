const std = @import("std");
const smi = @import("smi.zig");

/// One table per target: index, type, remote, the strings of the device, then the fields of its sample.
/// A device column that is empty for the whole table is left out.
pub fn write(writer: *std.Io.Writer, devices: []const smi.Device, snapshot: smi.Collector.Snapshot) !void {
    inline for (@typeInfo(smi.Target).@"enum".field_names) |name| {
        try writeTable(writer, devices, snapshot.samples, @field(smi.Target, name));
    }
}

fn writeTable(
    writer: *std.Io.Writer,
    devices: []const smi.Device,
    samples: []const smi.Sample,
    comptime target: smi.Target,
) !void {
    const fields = @typeInfo(@FieldType(smi.Sample, @tagName(target))).@"struct".field_names;

    var rows: usize = 0;
    var names = false;
    var drivers = false;
    var key_buffer: [16][]const u8 = undefined;
    var keys: std.ArrayList([]const u8) = .initBuffer(&key_buffer);
    for (devices) |device| {
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
    for (keys.items) |key| try writer.print(",{s}", .{key});
    inline for (fields) |field| try writer.writeAll("," ++ field);
    try writer.writeAll("\n");

    for (devices, samples, 0..) |device, captured, index| {
        if (device.target != target) continue;
        try writer.print("{d},{t},{any}", .{ index, target, device.remote });
        if (names) try writer.print(",{s}", .{device.name orelse ""});
        if (drivers) try writer.print(",{s}", .{device.driver_version orelse ""});
        for (keys.items) |key| try writer.print(",{s}", .{detailValue(device, key) orelse ""});

        const sample = @field(captured, @tagName(target));
        inline for (fields) |field| {
            try writer.writeAll(",");
            if (@field(sample, field)) |value| try writer.print("{d}", .{value});
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
