const std = @import("std");
const smi = @import("smi.zig");

pub const Row = struct {
    device: smi.Device,
    sample: smi.Sample,

    pub fn jsonStringify(self: Row, json: *std.json.Stringify) !void {
        std.debug.assert(std.meta.activeTag(self.device.info) == std.meta.activeTag(self.sample));

        try json.beginObject();
        try json.objectField(@tagName(self.device.info));
        try json.beginObject();
        try json.objectField("remote");
        try json.write(self.device.remote);
        try json.objectField("name");
        try json.write(self.device.name);

        switch (self.device.info) {
            inline else => |info| inline for (@typeInfo(@TypeOf(info)).@"struct".field_names) |name| {
                try json.objectField(name);
                try json.write(@field(info, name));
            },
        }

        switch (self.sample) {
            inline else => |fields| inline for (@typeInfo(@TypeOf(fields)).@"struct".field_names) |name| {
                try json.objectField(name);
                try json.write(@field(fields, name));
            },
        }

        try json.endObject();
        try json.endObject();
    }

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !Row {
        return jsonParseFromValue(allocator, try std.json.innerParse(std.json.Value, allocator, source, options), options);
    }

    pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) !Row {
        var lenient = options;
        lenient.ignore_unknown_fields = true;

        // Parse the same object three times (ignoring each other's fields) to keep parsing simple.
        // only used at startup, not on the poll loop.
        const info = try std.json.innerParseFromValue(smi.Device.Info, allocator, source, lenient);
        const sample = try std.json.innerParseFromValue(smi.Sample, allocator, source, lenient);
        const base = try std.json.innerParseFromValue(
            struct { remote: bool = false, name: ?[]const u8 = null },
            allocator,
            source.object.get(@tagName(info)).?,
            lenient,
        );

        return .{ .device = .{ .remote = base.remote, .name = base.name, .info = info }, .sample = sample };
    }
};

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
            try json.write(Row{ .device = device, .sample = self.collector.sample(@intCast(i)) });
        }

        try json.endArray();
    }
};
