const std = @import("std");
const smi = @import("smi.zig");

pub fn Response(comptime Devices: type) type {
    return struct {
        devices: Devices,
        processes: []const smi.ProcessInfo = &.{},
    };
}

pub fn write(writer: *std.Io.Writer, collector: *smi.Collector) !void {
    const processes = collector.processes.lock(collector.io);
    defer collector.processes.unlock(collector.io);

    const response: Response(Rows) = .{ .devices = .{ .collector = collector }, .processes = processes.rows.items };

    try std.json.Stringify.value(response, .{}, writer);
    try writer.writeAll("\n");
}

const Rows = struct {
    collector: *smi.Collector,

    pub fn jsonStringify(self: Rows, json: *std.json.Stringify) !void {
        try json.beginArray();

        for (self.collector.devices.items, 0..) |device, i| {
            try json.write(smi.Row{ .device = device, .sample = self.collector.sample(@fromBackingInt(@intCast(i))) });
        }

        try json.endArray();
    }
};
