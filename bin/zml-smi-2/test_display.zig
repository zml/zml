const std = @import("std");
const contract = @import("zml-smi-2/contract");
const tui = @import("zml-smi-2/tui");
const display = @import("display.zig");

const presentation = tui.presentation;

test "detail profiles render normalized device metrics and host power" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var samples: [5]contract.Metrics = @splat(.{});
    samples[0].put(.encoder_util_percent, 0);
    samples[0].put(.pcie_rx_kbps, 2_500);
    samples[1].put(.ambient_temperature, 43);
    samples[1].put(.liveness, 9);
    samples[1].put(.throttle_reasons, 1 << 48);
    samples[2].put(.power_mw, 0);
    samples[3].put(.nc_tensors, 3 << 20);

    const devices = try display.devices(allocator, &.{
        .{ .target = .cuda },   .{ .target = .furiosa }, .{ .target = .metal },
        .{ .target = .neuron }, .{ .target = .tpu },
    });
    const host: contract.HostData = .{ .cpu_power_mw = 3_250 };

    const gpu = devices[0].profile;
    const gpu_label = try presentation.detailLabel(allocator, &samples[0], gpu, .util);
    const gpu_cards = try presentation.cards(allocator, gpu.sections, &samples[0], host);
    try std.testing.expectEqualStrings("Enc 0%  Dec N/A", gpu_label.?);
    try std.testing.expectEqualStrings(" N/A", gpu_cards[1].lines[1].value);
    try std.testing.expectEqualStrings(" 2.5 MB/s", gpu_cards[1].lines[2].value);

    const npu = devices[1].profile;
    const npu_label = try presentation.detailLabel(allocator, &samples[1], npu, .temperature);
    const npu_cards = try presentation.cards(allocator, npu.sections, &samples[1], host);
    try std.testing.expectEqualStrings("Ambient 43\u{00b0}C", npu_label.?);
    try std.testing.expectEqualStrings(" 9", npu_cards[0].lines[0].value);
    try std.testing.expectEqualStrings(" bit 48", npu_cards[0].lines[2].value);

    const unified_cards = try presentation.cards(allocator, devices[2].profile.sections, &samples[2], host);
    try std.testing.expectEqualStrings(" 0.0W", unified_cards[1].lines[0].value);
    try std.testing.expectEqualStrings(" 3.2W", unified_cards[1].lines[2].value);

    const core = devices[3].profile;
    const core_cards = try presentation.cards(allocator, core.sections, &samples[3], host);
    try std.testing.expectEqual(.hidden, core.environment);
    try std.testing.expectEqualStrings(" 3 MB", core_cards[0].lines[0].value);

    const tpu = devices[4].profile;
    const tpu_cards = try presentation.cards(allocator, tpu.sections, &samples[4], host);
    try std.testing.expectEqual(.hidden, tpu.environment);
    try std.testing.expectEqualStrings("Duty Cycle", tpu.util_title);
    try std.testing.expectEqual(0, tpu_cards.len);
}

test "overview aligns memory and preserves the power units of each profile" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var samples: [2]contract.Metrics = @splat(.{});
    samples[0].put(.mem_used_bytes, 9 << 20);
    samples[0].put(.mem_total_bytes, 100 << 20);
    samples[0].put(.temperature, 37);
    samples[0].put(.power_mw, 35_256);
    samples[0].put(.power_limit_mw, 300_000);
    samples[1].put(.power_mw, 35_256);
    const state: tui.data.SystemState = .{
        .provider = undefined,
        .devices = try display.devices(allocator, &.{ .{ .target = .furiosa }, .{ .target = .metal } }),
        .frame = .{ .samples = &samples },
        .tui_refresh_rate = 100,
    };
    const npu = try presentation.overviewDevice(allocator, &state, 0);
    try std.testing.expectEqualStrings("  9 / 100 MB", npu.gauges[1].suffix.?);
    try std.testing.expectEqualStrings("37\u{00b0}C |  35 W", npu.gauges[0].suffix.?);
    const unified = try presentation.overviewDevice(allocator, &state, 1);
    try std.testing.expectEqualStrings("N/A | 35.2 W", unified.gauges[0].suffix.?);
}
