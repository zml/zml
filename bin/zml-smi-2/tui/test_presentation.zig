const std = @import("std");
const contract = @import("zml-smi-2/contract");
const data = @import("data.zig");
const presentation = @import("presentation.zig");
const spec = @import("spec.zig");

test "memory labels preserve raw ratios and gauges clamp them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const devices: []const spec.DeviceView = &.{.{
        .title = "GPU 0: Fixture",
        .heading = "Device 0: Fixture",
        .remote = false,
        .profile = &.{ .util_label = "GPU", .util_title = "GPU Utilization", .memory_prefix = "VRAM " },
    }};
    const max = std.math.maxInt(u64);
    const readings = [_]struct { used: ?u64, total: ?u64, label: []const u8, gauge: ?u8 }{
        .{ .used = null, .total = null, .label = "N/A", .gauge = null },
        .{ .used = null, .total = 1024, .label = "N/A", .gauge = null },
        .{ .used = 0, .total = 0, .label = "N/A", .gauge = null },
        .{ .used = 0, .total = 1024, .label = "0%", .gauge = 0 },
        .{ .used = max, .total = max, .label = "100%", .gauge = 100 },
        .{ .used = max, .total = 1, .label = "1844674407370955161500%", .gauge = 100 },
    };
    for (readings) |reading| {
        var sample: contract.Metrics = .{};
        sample.put(.mem_used_bytes, reading.used);
        sample.put(.mem_total_bytes, reading.total);
        const state: data.SystemState = .{
            .provider = undefined,
            .devices = devices,
            .frame = .{ .samples = &.{sample} },
            .tui_refresh_rate = 100,
        };

        const chart = try presentation.historyChart(allocator, &state, 0, .memory);
        try std.testing.expectEqualStrings(reading.label, chart.value_label.?);
        const overview = try presentation.overviewDevice(allocator, &state, 0);
        try std.testing.expectEqual(reading.gauge, overview.gauges[1].value);
    }
}
