const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const ProcessInfo = smi_info.process_info.ProcessInfo;

pub fn write(writer: *std.Io.Writer, devices: []const *DeviceInfo, processes: []const ProcessInfo) !void {
    var jw: std.json.Stringify = .{ .writer = writer };

    try jw.beginObject();

    try jw.objectField("devices");
    try jw.beginArray();
    for (devices, 0..) |dev, i| {
        inline for (@typeInfo(DeviceInfo).@"union".fields) |tp| {
            const tag = @field(smi_info.Target, tp.name);

            switch (dev.*) {
                tag => |*db| {
                    try jw.write(Envelope(tp.name, tp.type.Value){ .index = i, .type = tp.name, .inner = db.front().* });
                },
                else => {},
            }
        }
    }
    try jw.endArray();

    try jw.objectField("processes");
    try jw.beginArray();
    for (processes) |proc| {
        try jw.write(proc);
    }
    try jw.endArray();

    try jw.endObject();
    try writer.writeAll("\n");
}

fn Envelope(comptime _: []const u8, comptime Inner: type) type {
    return struct {
        index: usize,
        type: []const u8,
        inner: Inner,

        pub fn jsonStringify(self: @This(), jw: *std.json.Stringify) !void {
            try jw.beginObject();

            try jw.objectField("index");
            try jw.write(self.index);
            try jw.objectField("type");
            try jw.write(self.type);

            inline for (@typeInfo(Inner).@"struct".fields) |f| {
                try jw.objectField(f.name);
                try jw.write(@field(self.inner, f.name));
            }

            try jw.endObject();
        }
    };
}
