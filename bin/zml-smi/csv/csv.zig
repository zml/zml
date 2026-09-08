const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;

pub fn write(writer: *std.Io.Writer, devices: []const *DeviceInfo) !void {
    const info = @typeInfo(DeviceInfo).@"union";
    inline for (info.field_names, info.field_types) |type_name, DeviceType| {
        const tag = @field(smi_info.Target, type_name);
        var header_printed = false;

        for (devices, 0..) |dev, i| {
            switch (dev.*) {
                tag => |*db| {
                    if (!header_printed) {
                        try writer.writeAll("index,type");
                        inline for (@typeInfo(DeviceType.Value).@"struct".field_names) |field_name| {
                            try writer.writeAll("," ++ field_name);
                        }
                        try writer.writeAll("\n");

                        header_printed = true;
                    }

                    const val = db.front().*;

                    try writer.print("{d},{s}", .{ i, type_name });
                    inline for (@typeInfo(DeviceType.Value).@"struct".field_names) |field_name| {
                        try writer.writeAll(",");
                        try writeValue(writer, @field(val, field_name));
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
