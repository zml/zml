//! Prometheus text format using snapshot field names and units. For example,
//! power remains in milliwatts, temperature in Celsius, and host memory in KiB.
const std = @import("std");
const contract = @import("zml-smi-2/contract");
const Snapshot = @import("zml-smi-2/snapshot").Snapshot;

const device_prefix = "zml_smi_device_";
const host_prefix = "zml_smi_host_";

pub fn write(writer: *std.Io.Writer, snapshot: Snapshot) std.Io.Writer.Error!void {
    if (snapshot.devices.len != 0) try gauge(writer, device_prefix, "info");
    for (snapshot.devices, 0..) |device, index| {
        try deviceLabels(writer, "info", device, index);
        if (device.driver_version) |version| try label(writer, "driver_version", version);
        try writer.writeAll("} 1\n");
    }

    var has_details = false;
    for (snapshot.devices, 0..) |device, index| {
        for (device.details, 0..) |detail, position| {
            if (!has_details) try gauge(writer, device_prefix, "detail_info");
            has_details = true;
            try deviceLabels(writer, "detail_info", device, index);
            // Position keeps duplicate inventory entries distinct.
            try writer.print(",detail=\"{d}\"", .{position});
            try label(writer, "key", detail.key);
            try label(writer, "value", detail.value);
            try writer.writeAll("} 1\n");
        }
    }

    for (std.enums.values(contract.Metric)) |metric| {
        const field = @tagName(metric);
        var has_values = false;
        for (snapshot.devices, snapshot.samples, 0..) |device, *sample, index| {
            const value = sample.get(metric) orelse continue;
            if (!has_values) try gauge(writer, device_prefix, field);
            has_values = true;
            try deviceLabels(writer, field, device, index);
            try writer.print("}} {d}\n", .{value});
        }
    }

    const host_fields = @typeInfo(contract.HostData).@"struct";
    inline for (host_fields.field_names, host_fields.field_types) |field, Field| {
        const inner = switch (@typeInfo(Field)) {
            .optional => |optional| optional.child,
            else => continue,
        };
        switch (@typeInfo(inner)) {
            .int, .float => if (@field(snapshot.host, field)) |value| {
                try gauge(writer, host_prefix, field);
                try writer.print(host_prefix ++ field ++ " {d}\n", .{value});
            },
            else => {},
        }
    }
}

fn gauge(writer: *std.Io.Writer, comptime prefix: []const u8, field: []const u8) !void {
    try writer.print("# TYPE " ++ prefix ++ "{s} gauge\n", .{field});
}

fn deviceLabels(writer: *std.Io.Writer, field: []const u8, device: contract.Device, index: usize) !void {
    try writer.print(device_prefix ++ "{s}{{device=\"{d}\",target=\"{t}\"", .{
        field, index, device.target,
    });
    if (device.name) |name| try label(writer, "name", name);
}

fn label(writer: *std.Io.Writer, comptime name: []const u8, value: []const u8) !void {
    try writer.writeAll("," ++ name ++ "=\"");
    for (value) |byte| switch (byte) {
        '\\' => try writer.writeAll("\\\\"),
        '"' => try writer.writeAll("\\\""),
        '\n' => try writer.writeAll("\\n"),
        else => try writer.writeByte(byte),
    };
    try writer.writeByte('"');
}
