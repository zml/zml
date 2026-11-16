const std = @import("std");
const smi = @import("smi.zig");

/// Every device with its latest sample, then the merged processes.
pub fn write(
    writer: *std.Io.Writer,
    devices: []const smi.Device,
    snapshot: smi.Collector.Snapshot,
) !void {
    var json: std.json.Stringify = .{ .writer = writer };
    try json.beginObject();
    try json.objectField("devices");
    try json.beginArray();
    for (devices, snapshot.samples) |device, sample| {
        try json.write(smi.Row{ .device = device, .sample = sample });
    }
    try json.endArray();

    try json.objectField("processes");
    try json.write(snapshot.processes);
    try json.endObject();
    try writer.writeAll("\n");
}
