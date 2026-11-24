const std = @import("std");
const smi = @import("smi.zig");

const Metric = struct { name: []const u8, help: []const u8, decimals: u8 = 0, factor: u64 = 1 };

fn Metrics(comptime T: type) type {
    return std.enums.EnumFieldStruct(std.meta.FieldEnum(T), ?Metric, null);
}

const gpu: Metrics(smi.Sample.Gpu) = .{
    .util_percent = .{ .name = "zml_gpu_utilization_percent", .help = "GPU utilization percentage" },
    .encoder_util_percent = .{ .name = "zml_gpu_encoder_utilization_percent", .help = "GPU encoder utilization percentage" },
    .decoder_util_percent = .{ .name = "zml_gpu_decoder_utilization_percent", .help = "GPU decoder utilization percentage" },
    .power_mw = .{ .name = "zml_gpu_power_watts", .help = "GPU power draw in watts", .decimals = 3 },
    .power_limit_mw = .{ .name = "zml_gpu_power_limit_watts", .help = "GPU power limit in watts", .decimals = 3 },
    .temperature = .{ .name = "zml_gpu_temperature_celsius", .help = "GPU temperature in celsius" },
    .fan_speed_percent = .{ .name = "zml_gpu_fan_speed_percent", .help = "GPU fan speed percentage" },
    .clock_graphics_mhz = .{ .name = "zml_gpu_clock_graphics_mhz", .help = "GPU graphics clock frequency in MHz" },
    .clock_sm_mhz = .{ .name = "zml_gpu_clock_sm_mhz", .help = "GPU SM clock frequency in MHz" },
    .clock_soc_mhz = null,
    .clock_mem_mhz = .{ .name = "zml_gpu_clock_memory_mhz", .help = "GPU memory clock frequency in MHz" },
    .clock_graphics_max_mhz = null,
    .clock_mem_max_mhz = null,
    .mem_used_bytes = .{ .name = "zml_gpu_memory_used_bytes", .help = "GPU memory used in bytes" },
    .mem_total_bytes = .{ .name = "zml_gpu_memory_total_bytes", .help = "GPU memory total in bytes" },
    .mem_bus_width = null,
    .pcie_tx_kbps = .{ .name = "zml_gpu_pcie_tx_bytes_per_second", .help = "GPU PCIe TX throughput in bytes per second", .factor = 1000 },
    .pcie_rx_kbps = .{ .name = "zml_gpu_pcie_rx_bytes_per_second", .help = "GPU PCIe RX throughput in bytes per second", .factor = 1000 },
    .pcie_bandwidth_mbps = null,
    .pcie_link_gen = null,
    .pcie_link_width = null,
};

const neuron: Metrics(smi.Sample.Neuron) = .{
    .util_percent = .{ .name = "zml_neuron_utilization_percent", .help = "Neuron core utilization percentage" },
    .mem_used_bytes = .{ .name = "zml_neuron_memory_used_bytes", .help = "Neuron core memory used in bytes" },
    .mem_total_bytes = .{ .name = "zml_neuron_memory_total_bytes", .help = "Neuron core memory total in bytes" },
    .nc_tensors = .{ .name = "zml_neuron_tensors_bytes", .help = "Neuron core tensor memory in bytes" },
    .nc_constants = .{ .name = "zml_neuron_constants_bytes", .help = "Neuron core constants memory in bytes" },
    .nc_model_code = .{ .name = "zml_neuron_model_code_bytes", .help = "Neuron core model code memory in bytes" },
    .nc_shared_scratchpad = .{ .name = "zml_neuron_shared_scratchpad_bytes", .help = "Neuron core shared scratchpad memory in bytes" },
    .nc_nonshared_scratchpad = .{ .name = "zml_neuron_nonshared_scratchpad_bytes", .help = "Neuron core non-shared scratchpad memory in bytes" },
    .nc_runtime = .{ .name = "zml_neuron_runtime_bytes", .help = "Neuron core runtime memory in bytes" },
    .nc_driver = .{ .name = "zml_neuron_driver_bytes", .help = "Neuron core driver memory in bytes" },
    .nc_dma_rings = .{ .name = "zml_neuron_dma_rings_bytes", .help = "Neuron core DMA rings memory in bytes" },
    .nc_collectives = .{ .name = "zml_neuron_collectives_bytes", .help = "Neuron core collectives memory in bytes" },
    .nc_notifications = .{ .name = "zml_neuron_notifications_bytes", .help = "Neuron core notifications memory in bytes" },
    .nc_uncategorized = .{ .name = "zml_neuron_uncategorized_bytes", .help = "Neuron core uncategorized memory in bytes" },
};

const tpu: Metrics(smi.Sample.Tpu) = .{
    .util_percent = .{ .name = "zml_tpu_utilization_percent", .help = "TPU utilization percentage" },
    .mem_used_bytes = .{ .name = "zml_tpu_memory_used_bytes", .help = "TPU memory used in bytes" },
    .mem_total_bytes = .{ .name = "zml_tpu_memory_total_bytes", .help = "TPU memory total in bytes" },
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

const groups = .{ smi.Sample.Gpu, smi.Sample.Neuron, smi.Sample.Tpu };

fn metricsOf(comptime T: type) Metrics(T) {
    return switch (T) {
        smi.Sample.Gpu => gpu,
        smi.Sample.Neuron => neuron,
        smi.Sample.Tpu => tpu,
        else => @compileError("no metrics for " ++ @typeName(T)),
    };
}

comptime {
    for (@typeInfo(smi.Sample).@"union".field_types) |T| {
        for (groups) |G| {
            if (G == T) break;
        } else @compileError("missing prometheus metrics for " ++ @typeName(T));
    }
}

pub fn write(writer: *std.Io.Writer, collector: *smi.Collector) !void {
    const devices = collector.devices.items;

    try writer.writeAll("# HELP zml_device_info Device metadata\n# TYPE zml_device_info gauge\n");
    for (devices, 0..) |device, i| {
        try writer.print("zml_device_info{{device=\"{d}\",type=\"{t}\"", .{ i, device.target });
        if (device.name) |name| try label(writer, "name", name);
        if (device.driver_version) |driver| try label(writer, "driver", driver);
        details: for (device.details, 0..) |detail, detail_index| {
            if (!isValidLabelName(detail.key)) continue;
            for (device.details[0..detail_index]) |prev| {
                if (std.mem.eql(u8, prev.key, detail.key)) continue :details;
            }
            try label(writer, detail.key, detail.value);
        }
        try writer.writeAll("} 1\n");
    }
    try writer.writeAll("\n");

    inline for (groups) |T| {
        try writeGroup(writer, collector, T, metricsOf(T));
    }

    const data = collector.host.read(collector.io);
    inline for (@typeInfo(smi.HostData).@"struct".field_names) |field| {
        if (comptime @field(host, field)) |metric| {
            if (@field(data, field)) |value| {
                try header(writer, metric);
                try writer.print("{s} ", .{metric.name});
                try number(writer, metric, value);
                try writer.writeAll("\n\n");
            }
        }
    }
}

fn writeGroup(writer: *std.Io.Writer, collector: *smi.Collector, comptime T: type, comptime metrics: Metrics(T)) !void {
    inline for (@typeInfo(T).@"struct".field_names) |field| {
        if (comptime @field(metrics, field)) |metric| {
            var written = false;
            for (collector.devices.items, 0..) |device, i| {
                const raw = switch (collector.sample(@intCast(i))) {
                    inline else => |fields| if (@TypeOf(fields) == T) @field(fields, field) else null,
                } orelse continue;
                const value: u64 = switch (@typeInfo(@TypeOf(raw))) {
                    .@"enum" => @backingInt(raw),
                    else => raw,
                };
                try writeSample(writer, metric, &written, i, device, value);
            }
            if (written) try writer.writeAll("\n");
        }
    }
}

fn writeSample(writer: *std.Io.Writer, metric: Metric, written: *bool, index: usize, device: smi.Device, value: u64) !void {
    if (!written.*) {
        try header(writer, metric);
        written.* = true;
    }
    try writer.print("{s}{{device=\"{d}\",type=\"{t}\",name=\"", .{ metric.name, index, device.target });
    try std.zig.stringEscape(device.name orelse "", writer);
    try writer.writeAll("\"} ");
    try number(writer, metric, value);
    try writer.writeAll("\n");
}

fn header(writer: *std.Io.Writer, metric: Metric) !void {
    try writer.print("# HELP {s} {s}\n# TYPE {s} gauge\n", .{ metric.name, metric.help, metric.name });
}

fn label(writer: *std.Io.Writer, name: []const u8, value: []const u8) !void {
    try writer.print(",{s}=\"", .{name});
    try std.zig.stringEscape(value, writer);
    try writer.writeAll("\"");
}

fn isValidLabelName(name: []const u8) bool {
    if (name.len == 0) return false;
    if (!std.ascii.isAlphabetic(name[0]) and name[0] != '_') return false;
    for (name[1..]) |ch| {
        if (!std.ascii.isAlphanumeric(ch) and ch != '_') return false;
    }

    return !std.mem.eql(u8, name, "device") and
        !std.mem.eql(u8, name, "type") and
        !std.mem.eql(u8, name, "name") and
        !std.mem.eql(u8, name, "driver");
}

fn number(writer: *std.Io.Writer, metric: Metric, value: anytype) !void {
    if (@TypeOf(value) == f32) return writer.printFloat(value, .{ .mode = .decimal, .precision = metric.decimals });
    const scaled = value *| metric.factor;
    if (metric.decimals == 0) return writer.printInt(scaled, 10, .lower, .{});
    const one = std.math.pow(u64, 10, metric.decimals);
    try writer.printInt(scaled / one, 10, .lower, .{});
    try writer.writeByte('.');
    try writer.printInt(scaled % one, 10, .lower, .{ .width = metric.decimals, .fill = '0', .alignment = .right });
}

test "prometheus label validation" {
    try std.testing.expect(isValidLabelName("cuda_version"));
    try std.testing.expect(isValidLabelName("_custom0"));
    try std.testing.expect(!isValidLabelName(""));
    try std.testing.expect(!isValidLabelName("0bad"));
    try std.testing.expect(!isValidLabelName("bad-key"));
    try std.testing.expect(!isValidLabelName("x=\"1\"} 1\nfake"));
    try std.testing.expect(!isValidLabelName("name"));
    try std.testing.expect(!isValidLabelName("driver"));
}

test "prometheus metric names are unique" {
    var buffer: [128][]const u8 = undefined;
    var names: std.ArrayList([]const u8) = .initBuffer(&buffer);
    inline for (groups) |T| {
        const map = comptime metricsOf(T);
        inline for (@typeInfo(T).@"struct".field_names) |field| {
            if (comptime @field(map, field)) |metric| {
                for (names.items) |prev| try std.testing.expect(!std.mem.eql(u8, prev, metric.name));
                try names.appendBounded(metric.name);
            }
        }
    }
    inline for (@typeInfo(smi.HostData).@"struct".field_names) |field| {
        if (comptime @field(host, field)) |metric| {
            for (names.items) |prev| try std.testing.expect(!std.mem.eql(u8, prev, metric.name));
            try names.appendBounded(metric.name);
        }
    }
}

test "prometheus number formatting" {
    var buf: [64]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);

    try number(&w, .{ .name = "", .help = "" }, @as(u64, 42));
    try std.testing.expectEqualStrings("42", w.buffered());

    w = .fixed(&buf);
    try number(&w, .{ .name = "", .help = "", .factor = 1000 }, @as(u64, 42));
    try std.testing.expectEqualStrings("42000", w.buffered());

    w = .fixed(&buf);
    try number(&w, .{ .name = "", .help = "", .decimals = 3 }, @as(u64, 5));
    try std.testing.expectEqualStrings("0.005", w.buffered());

    w = .fixed(&buf);
    try number(&w, .{ .name = "", .help = "", .decimals = 3 }, @as(u64, 12500));
    try std.testing.expectEqualStrings("12.500", w.buffered());

    w = .fixed(&buf);
    try number(&w, .{ .name = "", .help = "", .decimals = 2 }, @as(f32, 1.5));
    try std.testing.expectEqualStrings("1.50", w.buffered());
}
