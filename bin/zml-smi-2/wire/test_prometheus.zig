const std = @import("std");
const contract = @import("zml-smi-2/contract");
const prometheus = @import("prometheus.zig");

test "gauges use snapshot names and units and omit missing metrics" {
    const devices = [_]contract.Device{
        .{ .target = .cuda, .name = "GPU" },
        .{ .target = .furiosa },
    };
    var samples = [_]contract.Metrics{ .{}, .{} };
    samples[0].put(.util_percent, 0);
    samples[0].put(.power_mw, 12_345);
    samples[0].put(.clock_soc_mhz, 500);
    samples[0].put(.pcie_tx_kbps, 7);
    samples[1].put(.util_percent, 50);
    samples[1].put(.ambient_temperature, 0);

    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try prometheus.write(&output.writer, .{
        .devices = &devices,
        .samples = &samples,
        .processes = &.{.{ .pid = 42, .username = "worker" }},
        .host = .{
            .hostname = "worker-host",
            .kernel = "worker-kernel",
            .cpu_name = "worker-cpu",
            .cpu_cores = 0,
            .mem_total_kib = 4096,
            .mem_available_kib = 2,
            .load_1 = 1.25,
            .load_5 = 0.625,
            .load_15 = 3.5,
            .uptime_seconds = 123,
            .cpu_power_mw = 456,
            .system_power_mw = 1_001,
        },
    });

    try std.testing.expectEqualStrings(
        \\# TYPE zml_smi_device_info gauge
        \\zml_smi_device_info{device="0",target="cuda",name="GPU"} 1
        \\zml_smi_device_info{device="1",target="furiosa"} 1
        \\# TYPE zml_smi_device_util_percent gauge
        \\zml_smi_device_util_percent{device="0",target="cuda",name="GPU"} 0
        \\zml_smi_device_util_percent{device="1",target="furiosa"} 50
        \\# TYPE zml_smi_device_power_mw gauge
        \\zml_smi_device_power_mw{device="0",target="cuda",name="GPU"} 12345
        \\# TYPE zml_smi_device_clock_soc_mhz gauge
        \\zml_smi_device_clock_soc_mhz{device="0",target="cuda",name="GPU"} 500
        \\# TYPE zml_smi_device_pcie_tx_kbps gauge
        \\zml_smi_device_pcie_tx_kbps{device="0",target="cuda",name="GPU"} 7
        \\# TYPE zml_smi_device_ambient_temperature gauge
        \\zml_smi_device_ambient_temperature{device="1",target="furiosa"} 0
        \\# TYPE zml_smi_host_cpu_cores gauge
        \\zml_smi_host_cpu_cores 0
        \\# TYPE zml_smi_host_mem_total_kib gauge
        \\zml_smi_host_mem_total_kib 4096
        \\# TYPE zml_smi_host_mem_available_kib gauge
        \\zml_smi_host_mem_available_kib 2
        \\# TYPE zml_smi_host_load_1 gauge
        \\zml_smi_host_load_1 1.25
        \\# TYPE zml_smi_host_load_5 gauge
        \\zml_smi_host_load_5 0.625
        \\# TYPE zml_smi_host_load_15 gauge
        \\zml_smi_host_load_15 3.5
        \\# TYPE zml_smi_host_uptime_seconds gauge
        \\zml_smi_host_uptime_seconds 123
        \\# TYPE zml_smi_host_cpu_power_mw gauge
        \\zml_smi_host_cpu_power_mw 456
        \\# TYPE zml_smi_host_system_power_mw gauge
        \\zml_smi_host_system_power_mw 1001
        \\
    , output.written());
}

test "every contract metric has a gauge" {
    const fields = std.enums.values(contract.Metric);
    var sample: contract.Metrics = .{};
    for (fields, 1..) |field, value| sample.put(field, value);

    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try prometheus.write(&output.writer, .{
        .devices = &.{.{ .target = .metal }},
        .samples = &.{sample},
        .processes = &.{},
        .host = .{},
    });

    var lines = std.mem.splitScalar(u8, output.written(), '\n');
    try std.testing.expectEqualStrings("# TYPE zml_smi_device_info gauge", lines.next().?);
    try std.testing.expectEqualStrings(
        "zml_smi_device_info{device=\"0\",target=\"metal\"} 1",
        lines.next().?,
    );
    var buffer: [160]u8 = undefined;
    for (fields, 1..) |field, value| {
        const name = @tagName(field);
        const declaration = try std.fmt.bufPrint(&buffer, "# TYPE zml_smi_device_{s} gauge", .{name});
        try std.testing.expectEqualStrings(declaration, lines.next().?);
        const reading = try std.fmt.bufPrint(
            &buffer,
            "zml_smi_device_{s}{{device=\"0\",target=\"metal\"}} {d}",
            .{ name, value },
        );
        try std.testing.expectEqualStrings(reading, lines.next().?);
    }
    try std.testing.expectEqualStrings("", lines.next().?);
    try std.testing.expectEqual(null, lines.next());
}

test "inventory details escape text and keep arbitrary keys in label values" {
    const detail: contract.Device.Detail = .{ .key = "slot\t#\r\n\"\\", .value = "hi\n\"\\ μ" };
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try prometheus.write(&output.writer, .{
        .devices = &.{.{
            .target = .tpu,
            .name = "N\"PU",
            .driver_version = "major\\minor",
            .details = &.{ detail, detail },
        }},
        .samples = &.{.{}},
        .processes = &.{},
        .host = .{},
    });

    const labels = "device=\"0\",target=\"tpu\",name=\"N\\\"PU\"";
    const detail_labels = ",key=\"slot\t#\r\\n\\\"\\\\\",value=\"hi\\n\\\"\\\\ μ\"} 1\n";
    try std.testing.expectEqualStrings(
        "# TYPE zml_smi_device_info gauge\n" ++
            "zml_smi_device_info{" ++ labels ++ ",driver_version=\"major\\\\minor\"} 1\n" ++
            "# TYPE zml_smi_device_detail_info gauge\n" ++
            "zml_smi_device_detail_info{" ++ labels ++ ",detail=\"0\"" ++ detail_labels ++
            "zml_smi_device_detail_info{" ++ labels ++ ",detail=\"1\"" ++ detail_labels,
        output.written(),
    );
}
