const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;

pub fn write(writer: *std.Io.Writer, devices: []const *DeviceInfo) !void {
    inline for (@typeInfo(DeviceInfo).@"union".fields) |tp| {
        const tag = @field(smi_info.Target, tp.name);
        var header_printed = false;

        for (devices, 0..) |dev, i| {
            switch (dev.*) {
                tag => |*db| {
                    if (!header_printed) {
                        try writer.writeAll("index,type");
                        inline for (@typeInfo(tp.type.Value).@"struct".fields) |f| {
                            try writer.writeAll("," ++ f.name);
                        }
                        try writer.writeAll("\n");

                        header_printed = true;
                    }

                    const val = db.front().*;

                    try writer.print("{d},{s}", .{ i, tp.name });
                    inline for (@typeInfo(tp.type.Value).@"struct".fields) |f| {
                        try writer.writeAll(",");
                        try writeValue(writer, @field(val, f.name));
                    }
                    try writer.writeAll("\n");
                },
                else => {},
            }
        }
    }
}

fn writeValue(writer: *std.Io.Writer, field: anytype) !void {
    if (@typeInfo(@TypeOf(field)) == .optional) {
        if (field) |v| return writeValue(writer, v);
    } else switch (@typeInfo(@TypeOf(field))) {
        .pointer => return writer.writeAll(field),
        else => return writer.print("{any}", .{field}),
    }
}
