const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const HostInfo = smi_info.host_info.HostInfo;

const gpu_metrics = .{
    .{ .field = "util_percent", .metric = "zml_device_utilization_percent", .help = "GPU utilization percentage" },
    .{ .field = "encoder_util_percent", .metric = "zml_device_encoder_utilization_percent", .help = "Encoder utilization percentage" },
    .{ .field = "decoder_util_percent", .metric = "zml_device_decoder_utilization_percent", .help = "Decoder utilization percentage" },
    .{ .field = "power_mw", .metric = "zml_device_power_watts", .help = "Power draw in watts", .fmt = fmtMilliwattsAsWatts },
    .{ .field = "power_limit_mw", .metric = "zml_device_power_limit_watts", .help = "Power limit in watts", .fmt = fmtMilliwattsAsWatts },
    .{ .field = "temperature", .metric = "zml_device_temperature_celsius", .help = "Device temperature in celsius" },
    .{ .field = "fan_speed_percent", .metric = "zml_device_fan_speed_percent", .help = "Fan speed percentage" },
    .{ .field = "clock_graphics_mhz", .metric = "zml_device_clock_graphics_mhz", .help = "Graphics clock frequency in MHz" },
    .{ .field = "clock_sm_mhz", .metric = "zml_device_clock_sm_mhz", .help = "SM clock frequency in MHz" },
    .{ .field = "clock_mem_mhz", .metric = "zml_device_clock_memory_mhz", .help = "Memory clock frequency in MHz" },
    .{ .field = "mem_used_bytes", .metric = "zml_device_memory_used_bytes", .help = "Device memory used in bytes" },
    .{ .field = "mem_total_bytes", .metric = "zml_device_memory_total_bytes", .help = "Device memory total in bytes" },
    .{ .field = "pcie_tx_kbps", .metric = "zml_device_pcie_tx_bytes_per_second", .help = "PCIe TX throughput in bytes per second", .fmt = fmtKbpsAsBytesPerSec },
    .{ .field = "pcie_rx_kbps", .metric = "zml_device_pcie_rx_bytes_per_second", .help = "PCIe RX throughput in bytes per second", .fmt = fmtKbpsAsBytesPerSec },
};

const neuron_metrics = .{
    .{ .field = "util_percent", .metric = "zml_device_utilization_percent", .help = "Device utilization percentage" },
    .{ .field = "mem_used_bytes", .metric = "zml_device_memory_used_bytes", .help = "Device memory used in bytes" },
    .{ .field = "mem_total_bytes", .metric = "zml_device_memory_total_bytes", .help = "Device memory total in bytes" },
    .{ .field = "nc_tensors", .metric = "zml_device_neuron_tensors_bytes", .help = "Neuron core tensor memory in bytes" },
    .{ .field = "nc_constants", .metric = "zml_device_neuron_constants_bytes", .help = "Neuron core constants memory in bytes" },
    .{ .field = "nc_model_code", .metric = "zml_device_neuron_model_code_bytes", .help = "Neuron core model code memory in bytes" },
    .{ .field = "nc_shared_scratchpad", .metric = "zml_device_neuron_shared_scratchpad_bytes", .help = "Neuron core shared scratchpad memory in bytes" },
    .{ .field = "nc_nonshared_scratchpad", .metric = "zml_device_neuron_nonshared_scratchpad_bytes", .help = "Neuron core non-shared scratchpad memory in bytes" },
    .{ .field = "nc_runtime", .metric = "zml_device_neuron_runtime_bytes", .help = "Neuron core runtime memory in bytes" },
    .{ .field = "nc_driver", .metric = "zml_device_neuron_driver_bytes", .help = "Neuron core driver memory in bytes" },
    .{ .field = "nc_dma_rings", .metric = "zml_device_neuron_dma_rings_bytes", .help = "Neuron core DMA rings memory in bytes" },
    .{ .field = "nc_collectives", .metric = "zml_device_neuron_collectives_bytes", .help = "Neuron core collectives memory in bytes" },
    .{ .field = "nc_notifications", .metric = "zml_device_neuron_notifications_bytes", .help = "Neuron core notifications memory in bytes" },
    .{ .field = "nc_uncategorized", .metric = "zml_device_neuron_uncategorized_bytes", .help = "Neuron core uncategorized memory in bytes" },
};

const tpu_metrics = .{
    .{ .field = "util_percent", .metric = "zml_device_utilization_percent", .help = "Device utilization percentage" },
    .{ .field = "mem_used_bytes", .metric = "zml_device_memory_used_bytes", .help = "Device memory used in bytes" },
    .{ .field = "mem_total_bytes", .metric = "zml_device_memory_total_bytes", .help = "Device memory total in bytes" },
};

const host_metrics = .{
    .{ .field = "cpu_cores", .metric = "zml_host_cpu_cores", .help = "Number of CPU cores" },
    .{ .field = "mem_total_kib", .metric = "zml_host_memory_total_bytes", .help = "Total host memory in bytes", .fmt = fmtKibAsBytes },
    .{ .field = "mem_available_kib", .metric = "zml_host_memory_available_bytes", .help = "Available host memory in bytes", .fmt = fmtKibAsBytes },
    .{ .field = "load_1", .metric = "zml_host_load_1", .help = "1-minute load average" },
    .{ .field = "load_5", .metric = "zml_host_load_5", .help = "5-minute load average" },
    .{ .field = "load_15", .metric = "zml_host_load_15", .help = "15-minute load average" },
    .{ .field = "uptime_seconds", .metric = "zml_host_uptime_seconds", .help = "System uptime in seconds" },
};

pub fn write(writer: *std.Io.Writer, devices: []const *DeviceInfo, host: *const HostInfo) !void {
    try writer.writeAll("# HELP zml_device_info Device metadata\n");
    try writer.writeAll("# TYPE zml_device_info gauge\n");
    for (devices, 0..) |dev, i| {
        inline for (@typeInfo(DeviceInfo).@"union".fields) |tp| {
            const tag = @field(smi_info.device_info.Target, tp.name);

            switch (dev.*) {
                tag => |*db| {
                    const val = db.front().*;

                    try writer.print("zml_device_info{{device=\"{d}\",type=\"{s}\"", .{ i, tp.name });

                    inline for (.{
                        .{ "name", "name" },
                        .{ "driver_version", "driver" },
                        .{ "cuda_driver_version", "cuda_version" },
                    }) |label| {
                        if (@hasField(@TypeOf(val), label[0])) {
                            if (@field(val, label[0])) |v| {
                                try writer.print(",{s}=\"{s}\"", .{ label[1], v });
                            }
                        }
                    }

                    try writer.writeAll("} 1\n");
                },
                else => {},
            }
        }
    }
    try writer.writeAll("\n");

    for (devices, 0..) |dev, i| {
        switch (dev.*) {
            .cuda => |*db| try writeMetrics(writer, gpu_metrics, db.front().*, .{ i, "cuda" }),
            .rocm => |*db| try writeMetrics(writer, gpu_metrics, db.front().*, .{ i, "rocm" }),
            .neuron => |*db| try writeMetrics(writer, neuron_metrics, db.front().*, .{ i, "neuron" }),
            .tpu => |*db| try writeMetrics(writer, tpu_metrics, db.front().*, .{ i, "tpu" }),
        }
    }

    try writeMetrics(writer, host_metrics, host.front().*, null);
}

fn writeMetrics(
    writer: *std.Io.Writer,
    comptime metrics: anytype,
    val: anytype,
    device: ?struct { usize, []const u8 },
) !void {
    const name_label = if (@hasField(@TypeOf(val), "name")) val.name else null;

    inline for (metrics) |m| {
        if (@field(val, m.field)) |raw| {
            try writer.print("# HELP {s} {s}\n# TYPE {s} gauge\n{s}", .{ m.metric, m.help, m.metric, m.metric });

            if (device) |d| {
                try writer.print("{{device=\"{d}\",type=\"{s}\"", .{ d[0], d[1] });

                if (name_label) |name| {
                    try writer.print(",name=\"{s}\"", .{name});
                }

                try writer.writeAll("}");
            }

            try writer.writeAll(" ");

            if (@hasField(@TypeOf(m), "fmt")) {
                try m.fmt(writer, raw);
            } else {
                try writeValue(writer, raw);
            }
            try writer.writeAll("\n");
        }
    }
}

fn writeValue(writer: *std.Io.Writer, val: anytype) std.Io.Writer.Error!void {
    switch (@typeInfo(@TypeOf(val))) {
        .float, .comptime_float => try writer.print("{d:.2}", .{val}),
        .pointer => try writer.writeAll(val),
        else => try writer.print("{d}", .{val}),
    }
}

fn fmtMilliwattsAsWatts(writer: *std.Io.Writer, val: u64) std.Io.Writer.Error!void {
    try writer.print("{d}.{d:0>3}", .{ val / 1000, val % 1000 });
}

fn fmtKibAsBytes(writer: *std.Io.Writer, val: u64) std.Io.Writer.Error!void {
    try writer.print("{d}", .{val * 1024});
}

fn fmtKbpsAsBytesPerSec(writer: *std.Io.Writer, val: u64) std.Io.Writer.Error!void {
    try writer.print("{d}", .{val * 128});
}
