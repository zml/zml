const std = @import("std");
const contract = @import("zml-smi-2/contract");
const Snapshot = @import("zml-smi-2/snapshot").Snapshot;

const metrics = std.enums.values(contract.Metric);

pub fn write(writer: *std.Io.Writer, snapshot: Snapshot) std.Io.Writer.Error!void {
    try writer.writeAll("index,target,remote,name,driver_version,details");
    for (metrics) |metric| try writer.print(",{t}", .{metric});
    try writer.writeByte('\n');

    var buffer: [512]u8 = undefined;
    var storage = std.heap.BufferFirstAllocator.init(&buffer, std.heap.page_allocator);
    var details: std.Io.Writer.Allocating = .init(storage.allocator());
    defer details.deinit();

    for (snapshot.devices, snapshot.samples, 0..) |device, sample, index| {
        try writer.print("{d},{t},{any}", .{ index, device.target, device.remote });
        try writeField(writer, device.name orelse "", false);
        try writeField(writer, device.driver_version orelse "", false);
        details.clearRetainingCapacity();
        try std.json.Stringify.value(device.details, .{
            .emit_null_optional_fields = false,
        }, &details.writer);
        try writeField(writer, details.written(), true);
        for (metrics) |metric| {
            try writer.writeByte(',');
            if (sample.get(metric)) |value| try writer.print("{d}", .{value});
        }
        try writer.writeByte('\n');
    }
}

fn writeField(writer: *std.Io.Writer, text: []const u8, force_quote: bool) !void {
    try writer.writeByte(',');
    if (!force_quote and std.mem.findAny(u8, text, ",\"\r\n") == null) return writer.writeAll(text);
    try writer.writeByte('"');
    var parts = std.mem.splitScalar(u8, text, '"');
    try writer.writeAll(parts.first());
    while (parts.next()) |part| {
        try writer.writeAll("\"\"");
        try writer.writeAll(part);
    }
    try writer.writeByte('"');
}
