const std = @import("std");
const smi = @import("smi.zig");

pub fn write(writer: *std.Io.Writer, collector: *smi.Collector) !void {
    var json: std.json.Stringify = .{ .writer = writer };
    try json.beginObject();
    try json.objectField("devices");
    try json.beginArray();
    for (collector.devices.items, 0..) |device, i| {
        try json.write(smi.Row{ .device = device, .sample = collector.sample(@fromBackingInt(@intCast(i))) });
    }
    try json.endArray();

    try json.objectField("processes");
    const processes = collector.processes.lock(collector.io);
    defer collector.processes.unlock(collector.io);
    try json.write(processes.rows.items);
    try json.endObject();
    try writer.writeAll("\n");
}
