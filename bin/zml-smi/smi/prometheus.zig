const std = @import("std");
const smi = @import("smi.zig");

/// A metric family. `decimals` moves the decimal point and `factor` multiplies, so conversions are exact.
const Metric = struct { name: []const u8, help: []const u8, decimals: u8 = 0, factor: u64 = 1 };

/// One entry per field of `T`. A missing or misspelled field does not compile.
fn Metrics(comptime T: type) type {
    return std.enums.EnumFieldStruct(std.meta.FieldEnum(T), ?Metric, null);
}

const utilization: Metric = .{ .name = "zml_device_utilization_percent", .help = "Device utilization percentage" };
const memory_used: Metric = .{ .name = "zml_device_memory_used_bytes", .help = "Device memory used in bytes" };
const memory_total: Metric = .{ .name = "zml_device_memory_total_bytes", .help = "Device memory total in bytes" };

const gpu: Metrics(smi.Gpu) = .{
    .util_percent = utilization,
    .encoder_util_percent = .{ .name = "zml_device_encoder_utilization_percent", .help = "Encoder utilization percentage" },
    .decoder_util_percent = .{ .name = "zml_device_decoder_utilization_percent", .help = "Decoder utilization percentage" },
    // Milliwatts, written as watts.
    .power_mw = .{ .name = "zml_device_power_watts", .help = "Power draw in watts", .decimals = 3 },
    .power_limit_mw = .{ .name = "zml_device_power_limit_watts", .help = "Power limit in watts", .decimals = 3 },
    .temperature = .{ .name = "zml_device_temperature_celsius", .help = "Device temperature in celsius" },
    .fan_speed_percent = .{ .name = "zml_device_fan_speed_percent", .help = "Fan speed percentage" },
    .clock_graphics_mhz = .{ .name = "zml_device_clock_graphics_mhz", .help = "Graphics clock frequency in MHz" },
    .clock_sm_mhz = .{ .name = "zml_device_clock_sm_mhz", .help = "SM clock frequency in MHz" },
    .clock_soc_mhz = null,
    .clock_mem_mhz = .{ .name = "zml_device_clock_memory_mhz", .help = "Memory clock frequency in MHz" },
    .clock_graphics_max_mhz = null,
    .clock_mem_max_mhz = null,
    .mem_used_bytes = memory_used,
    .mem_total_bytes = memory_total,
    .mem_bus_width = null,
    // Kilobytes per second, written as bytes per second.
    .pcie_tx_kbps = .{ .name = "zml_device_pcie_tx_bytes_per_second", .help = "PCIe TX throughput in bytes per second", .factor = 1000 },
    .pcie_rx_kbps = .{ .name = "zml_device_pcie_rx_bytes_per_second", .help = "PCIe RX throughput in bytes per second", .factor = 1000 },
    .pcie_bandwidth_mbps = null,
    .pcie_link_gen = null,
    .pcie_link_width = null,
};

const neuron: Metrics(smi.Neuron) = .{
    .util_percent = utilization,
    .mem_used_bytes = memory_used,
    .mem_total_bytes = memory_total,
    .nc_tensors = .{ .name = "zml_device_neuron_tensors_bytes", .help = "Neuron core tensor memory in bytes" },
    .nc_constants = .{ .name = "zml_device_neuron_constants_bytes", .help = "Neuron core constants memory in bytes" },
    .nc_model_code = .{ .name = "zml_device_neuron_model_code_bytes", .help = "Neuron core model code memory in bytes" },
    .nc_shared_scratchpad = .{ .name = "zml_device_neuron_shared_scratchpad_bytes", .help = "Neuron core shared scratchpad memory in bytes" },
    .nc_nonshared_scratchpad = .{ .name = "zml_device_neuron_nonshared_scratchpad_bytes", .help = "Neuron core non-shared scratchpad memory in bytes" },
    .nc_runtime = .{ .name = "zml_device_neuron_runtime_bytes", .help = "Neuron core runtime memory in bytes" },
    .nc_driver = .{ .name = "zml_device_neuron_driver_bytes", .help = "Neuron core driver memory in bytes" },
    .nc_dma_rings = .{ .name = "zml_device_neuron_dma_rings_bytes", .help = "Neuron core DMA rings memory in bytes" },
    .nc_collectives = .{ .name = "zml_device_neuron_collectives_bytes", .help = "Neuron core collectives memory in bytes" },
    .nc_notifications = .{ .name = "zml_device_neuron_notifications_bytes", .help = "Neuron core notifications memory in bytes" },
    .nc_uncategorized = .{ .name = "zml_device_neuron_uncategorized_bytes", .help = "Neuron core uncategorized memory in bytes" },
};

const tpu: Metrics(smi.Tpu) = .{
    .util_percent = utilization,
    .mem_used_bytes = memory_used,
    .mem_total_bytes = memory_total,
};

const furiosa: Metrics(smi.Furiosa) = .{
    .util_percent = utilization,
    .mem_used_bytes = memory_used,
    .mem_total_bytes = memory_total,
    .power_mw = gpu.power_mw,
    .power_limit_mw = gpu.power_limit_mw,
    .temperature = gpu.temperature,
    .ambient_temperature = .{ .name = "zml_device_ambient_temperature_celsius", .help = "Ambient temperature in celsius" },
    .governor = .{ .name = "zml_device_governor", .help = "Governor profile: 0 performance, 1 powersave" },
    .liveness = .{ .name = "zml_device_liveness", .help = "Liveness: 1 alive, 0 unknown" },
    .throttle_reasons = .{ .name = "zml_device_throttle_reasons", .help = "Active throttle reasons as a bitmask" },
};

const host: Metrics(smi.HostData) = .{
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
};

fn metricsOf(comptime T: type) Metrics(T) {
    return switch (T) {
        smi.Gpu => gpu,
        smi.Neuron => neuron,
        smi.Tpu => tpu,
        smi.Furiosa => furiosa,
        else => @compileError("no metrics for " ++ @typeName(T)),
    };
}

/// The distinct families of every struct, in the order the structs give them.
/// TPU first keeps the order of the families: utilization and memory, then GPU, then Neuron.
const families: []const Metric = families: {
    @setEvalBranchQuota(10_000);
    var list: []const Metric = &.{};
    for (.{ smi.Tpu, smi.Gpu, smi.Neuron, smi.Furiosa }) |T| {
        const metrics = metricsOf(T);
        next: for (@typeInfo(T).@"struct".field_names) |field| {
            const metric = @field(metrics, field) orelse continue;
            for (list) |seen| {
                if (!std.mem.eql(u8, seen.name, metric.name)) continue;
                if (!std.mem.eql(u8, seen.help, metric.help) or seen.decimals != metric.decimals or seen.factor != metric.factor) {
                    @compileError(metric.name ++ " has two definitions");
                }
                continue :next;
            }
            list = list ++ .{metric};
        }
    }
    break :families list;
};

/// The field of `T` written as `family`, or null.
fn fieldOf(comptime T: type, comptime family: Metric) ?[]const u8 {
    @setEvalBranchQuota(10_000);
    const metrics = metricsOf(T);
    for (@typeInfo(T).@"struct".field_names) |field| {
        const metric = @field(metrics, field) orelse continue;
        if (std.mem.eql(u8, metric.name, family.name)) return field;
    }
    return null;
}

pub fn write(writer: *std.Io.Writer, collector: *smi.Collector) !void {
    const devices = collector.devices.items;

    try writer.writeAll("# HELP zml_device_info Device metadata\n# TYPE zml_device_info gauge\n");
    for (devices, 0..) |device, i| {
        try writer.print("zml_device_info{{device=\"{d}\",type=\"{t}\"", .{ i, device.target });
        if (device.name) |name| try label(writer, "name", name);
        if (device.driver_version) |driver| try label(writer, "driver", driver);
        for (device.details) |detail| try label(writer, detail.key, detail.value);
        try writer.writeAll("} 1\n");
    }
    try writer.writeAll("\n");

    inline for (families) |family| {
        var written = false;
        for (devices, 0..) |device, i| {
            const value: u64 = switch (collector.sample(@fromBackingInt(@intCast(i)))) {
                inline else => |fields| if (comptime fieldOf(@TypeOf(fields), family)) |field| @field(fields, field) else null,
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

    const data = collector.host.read(collector.io);
    inline for (@typeInfo(smi.HostData).@"struct".field_names) |field| {
        if (comptime @field(host, field)) |family| {
            if (@field(data, field)) |value| {
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
