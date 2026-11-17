const std = @import("std");
const model = @import("zml-smi/model");
const prometheus = @import("zml-smi/formats").prometheus;

test "snapshot output preserves family order, missing values, labels and unit conversions" {
    const devices = [_]model.Device{
        .{
            .target = .cuda,
            .name = "GPU",
            .driver_version = "535",
            .details = &.{.{ .key = "cuda_version", .label = "CUDA", .value = "12.0" }},
        },
        .{ .target = .furiosa, .name = "N\"PU" },
    };
    const samples = [_]model.Sample{
        .{ .cuda = .{
            .util_percent = 0,
            .encoder_util_percent = 0,
            .power_mw = 12_345,
            .pcie_tx_kbps = 7,
        } },
        .{ .furiosa = .{
            .util_percent = 50,
            .power_mw = 1_000,
            .ambient_temperature = 0,
            .governor = 1,
            .throttle_reasons = 0,
        } },
    };
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try prometheus.write(&output.writer, .{
        .devices = &devices,
        .samples = &samples,
        .processes = &.{},
        .host = .{ .mem_available_kib = 2, .load_1 = 1.25, .system_power_mw = 1_001 },
    });

    try std.testing.expectEqualStrings(
        \\# HELP zml_device_info Device metadata
        \\# TYPE zml_device_info gauge
        \\zml_device_info{device="0",type="cuda",name="GPU",driver="535",cuda_version="12.0"} 1
        \\zml_device_info{device="1",type="furiosa",name="N\"PU"} 1
        \\
        \\# HELP zml_device_utilization_percent Device utilization percentage
        \\# TYPE zml_device_utilization_percent gauge
        \\zml_device_utilization_percent{device="0",type="cuda",name="GPU"} 0
        \\zml_device_utilization_percent{device="1",type="furiosa",name="N\"PU"} 50
        \\
        \\# HELP zml_device_encoder_utilization_percent Encoder utilization percentage
        \\# TYPE zml_device_encoder_utilization_percent gauge
        \\zml_device_encoder_utilization_percent{device="0",type="cuda",name="GPU"} 0
        \\
        \\# HELP zml_device_power_watts Power draw in watts
        \\# TYPE zml_device_power_watts gauge
        \\zml_device_power_watts{device="0",type="cuda",name="GPU"} 12.345
        \\zml_device_power_watts{device="1",type="furiosa",name="N\"PU"} 1.000
        \\
        \\# HELP zml_device_pcie_tx_bytes_per_second PCIe TX throughput in bytes per second
        \\# TYPE zml_device_pcie_tx_bytes_per_second gauge
        \\zml_device_pcie_tx_bytes_per_second{device="0",type="cuda",name="GPU"} 7000
        \\
        \\# HELP zml_device_ambient_temperature_celsius Ambient temperature in celsius
        \\# TYPE zml_device_ambient_temperature_celsius gauge
        \\zml_device_ambient_temperature_celsius{device="1",type="furiosa",name="N\"PU"} 0
        \\
        \\# HELP zml_device_governor Governor profile: 0 performance, 1 powersave
        \\# TYPE zml_device_governor gauge
        \\zml_device_governor{device="1",type="furiosa",name="N\"PU"} 1
        \\
        \\# HELP zml_device_throttle_reasons Active throttle reasons as a bitmask
        \\# TYPE zml_device_throttle_reasons gauge
        \\zml_device_throttle_reasons{device="1",type="furiosa",name="N\"PU"} 0
        \\
        \\# HELP zml_host_memory_available_bytes Available host memory in bytes
        \\# TYPE zml_host_memory_available_bytes gauge
        \\zml_host_memory_available_bytes 2048
        \\
        \\# HELP zml_host_load_1 1-minute load average
        \\# TYPE zml_host_load_1 gauge
        \\zml_host_load_1 1.25
        \\
        \\# HELP zml_host_system_power_watts System power draw in watts
        \\# TYPE zml_host_system_power_watts gauge
        \\zml_host_system_power_watts 1.001
        \\
        \\
    , output.written());
}
