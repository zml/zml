const std = @import("std");
const smi = @import("smi.zig");

/// Multiply by factor, then move the decimal point; conversions remain exact.
const Metric = struct {
    name: []const u8,
    help: []const u8,
    decimals: u8 = 0,
    factor: u64 = 1,
};

const DeviceMetric = struct { field: []const u8, metric: Metric };

/// The public family order is independent of internal sample organization.
const device_metrics = [_]DeviceMetric{
    deviceMetric("util_percent", "utilization_percent", "Device utilization percentage"),
    deviceMetric("mem_used_bytes", "memory_used_bytes", "Device memory used in bytes"),
    deviceMetric("mem_total_bytes", "memory_total_bytes", "Device memory total in bytes"),
    engine("encoder", "Encoder"),
    engine("decoder", "Decoder"),
    deviceMetric("power_mw", "power_watts", "Power draw in watts"),
    deviceMetric("power_limit_mw", "power_limit_watts", "Power limit in watts"),
    deviceMetric("temperature", "temperature_celsius", "Device temperature in celsius"),
    deviceMetric("fan_speed_percent", "fan_speed_percent", "Fan speed percentage"),
    clock("graphics", "Graphics"),
    clock("sm", "SM"),
    clock("mem", "Memory"),
    traffic("tx", "TX"),
    traffic("rx", "RX"),
    neuron("tensors", "tensor"),
    neuron("constants", "constants"),
    neuron("model_code", "model code"),
    neuron("shared_scratchpad", "shared scratchpad"),
    neuron("nonshared_scratchpad", "non-shared scratchpad"),
    neuron("runtime", "runtime"),
    neuron("driver", "driver"),
    neuron("dma_rings", "DMA rings"),
    neuron("collectives", "collectives"),
    neuron("notifications", "notifications"),
    neuron("uncategorized", "uncategorized"),
    deviceMetric(
        "ambient_temperature",
        "ambient_temperature_celsius",
        "Ambient temperature in celsius",
    ),
    deviceMetric("governor", "governor", "Governor profile: 0 performance, 1 powersave"),
    deviceMetric("liveness", "liveness", "Liveness: 1 alive, 0 unknown"),
    deviceMetric("throttle_reasons", "throttle_reasons", "Active throttle reasons as a bitmask"),
    deviceMetric("ane_power_mw", "ane_power_watts", "Neural Engine power draw in watts"),
};

fn deviceMetric(comptime field: []const u8, comptime name: []const u8, help: []const u8) DeviceMetric {
    return .{
        .field = field,
        .metric = .{
            .name = "zml_device_" ++ name,
            .help = help,
            // Field suffixes specify native units; exports use watts and bytes per second.
            .decimals = if (std.mem.endsWith(u8, field, "_mw")) 3 else 0,
            .factor = if (std.mem.endsWith(u8, field, "_kbps")) 1000 else 1,
        },
    };
}

fn engine(comptime name: []const u8, comptime title: []const u8) DeviceMetric {
    return deviceMetric(
        name ++ "_util_percent",
        name ++ "_utilization_percent",
        title ++ " utilization percentage",
    );
}

fn clock(comptime kind: []const u8, comptime title: []const u8) DeviceMetric {
    const name = if (std.mem.eql(u8, kind, "mem")) "memory" else kind;
    return deviceMetric("clock_" ++ kind ++ "_mhz", "clock_" ++ name ++ "_mhz", title ++
        " clock frequency in MHz");
}

fn traffic(comptime direction: []const u8, comptime title: []const u8) DeviceMetric {
    return deviceMetric("pcie_" ++ direction ++ "_kbps", "pcie_" ++ direction ++
        "_bytes_per_second", "PCIe " ++ title ++ " throughput in bytes per second");
}

fn neuron(comptime name: []const u8, comptime title: []const u8) DeviceMetric {
    return deviceMetric("nc_" ++ name, "neuron_" ++ name ++ "_bytes", "Neuron core " ++ title ++
        " memory in bytes");
}

comptime {
    for (device_metrics) |entry| {
        const exists = for (@typeInfo(smi.Target).@"enum".field_names) |target| {
            if (@hasField(@FieldType(smi.Sample, target), entry.field)) break true;
        } else false;
        if (!exists) @compileError("unknown exported metric field: " ++ entry.field);
    }
}

const HostMetrics = std.enums.EnumFieldStruct(std.meta.FieldEnum(smi.HostData), ?Metric, null);
const host: HostMetrics = .{
    .hostname = null,
    .kernel = null,
    .cpu_name = null,
    .cpu_cores = .{ .name = "zml_host_cpu_cores", .help = "Number of CPU cores" },
    .mem_total_kib = .{ .name = "zml_host_memory_total_bytes", .help = "Total host memory in bytes", .factor = 1024 },
    .mem_available_kib = .{ .name = "zml_host_memory_available_bytes", .help = "Available host memory in bytes", .factor = 1024 },
    .load_1 = .{ .name = "zml_host_load_1", .help = "1-minute load average", .decimals = 2 },
    .load_5 = .{ .name = "zml_host_load_5", .help = "5-minute load average", .decimals = 2 },
    .load_15 = .{ .name = "zml_host_load_15", .help = "15-minute load average", .decimals = 2 },
    .uptime_seconds = .{ .name = "zml_host_uptime_seconds", .help = "System uptime in seconds" },
    .cpu_power_mw = .{ .name = "zml_host_cpu_power_watts", .help = "CPU power draw in watts", .decimals = 3 },
    .system_power_mw = .{ .name = "zml_host_system_power_watts", .help = "System power draw in watts", .decimals = 3 },
};

pub fn write(writer: *std.Io.Writer, collector: *smi.Collector) !void {
    var frame: std.heap.ArenaAllocator = .init(collector.gpa);
    defer frame.deinit();
    const snapshot = try collector.capture(frame.allocator(), false);
    try writeSnapshot(writer, collector.devices.items, snapshot);
}

fn writeSnapshot(
    writer: *std.Io.Writer,
    devices: []const smi.Device,
    snapshot: smi.Collector.Snapshot,
) !void {
    try writer.writeAll("# HELP zml_device_info Device metadata\n# TYPE zml_device_info gauge\n");
    for (devices, 0..) |device, i| {
        try writer.print("zml_device_info{{device=\"{d}\",type=\"{t}\"", .{ i, device.target });
        if (device.name) |name| try label(writer, "name", name);
        if (device.driver_version) |driver| try label(writer, "driver", driver);
        for (device.details) |detail| try label(writer, detail.key, detail.value);
        try writer.writeAll("} 1\n");
    }
    try writer.writeAll("\n");

    inline for (device_metrics) |entry| {
        const family = entry.metric;
        var written = false;
        for (devices, 0..) |device, i| {
            const value: u64 = switch (snapshot.samples[i]) {
                inline else => |fields| if (@hasField(@TypeOf(fields), entry.field))
                    @field(fields, entry.field)
                else
                    null,
            } orelse continue;
            if (!written) try header(writer, family);
            written = true;
            try writer.print("{s}{{device=\"{d}\",type=\"{t}\",name=\"", .{ family.name, i, device.target });
            try std.zig.stringEscape(device.name orelse "", writer);
            try writer.writeAll("\"} ");
            try number(writer, family, value);
            try writer.writeAll("\n");
        }
        if (written) try writer.writeAll("\n");
    }

    inline for (@typeInfo(smi.HostData).@"struct".field_names) |field| {
        if (comptime @field(host, field)) |family| {
            if (@field(snapshot.host, field)) |value| {
                try header(writer, family);
                try writer.print("{s} ", .{family.name});
                try number(writer, family, value);
                try writer.writeAll("\n\n");
            }
        }
    }
}

fn header(writer: *std.Io.Writer, comptime family: Metric) !void {
    try writer.writeAll("# HELP " ++ family.name ++ " " ++ family.help ++ "\n# TYPE " ++ family.name ++ " gauge\n");
}

fn label(writer: *std.Io.Writer, name: []const u8, value: []const u8) !void {
    try writer.print(",{s}=\"", .{name});
    try std.zig.stringEscape(value, writer);
    try writer.writeAll("\"");
}

fn number(writer: *std.Io.Writer, comptime family: Metric, value: anytype) !void {
    if (@TypeOf(value) == f32) return writer.print(std.fmt.comptimePrint("{{d:.{d}}}", .{family.decimals}), .{value});
    const scaled = value *| family.factor;
    if (family.decimals == 0) return writer.print("{d}", .{scaled});
    const one = comptime std.math.pow(u64, 10, family.decimals);
    try writer.print(std.fmt.comptimePrint("{{d}}.{{d:0>{d}}}", .{family.decimals}), .{ scaled / one, scaled % one });
}

pub const testing = if (@import("builtin").is_test) struct {
    pub const snapshot = writeSnapshot;
} else struct {};
