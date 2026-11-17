const std = @import("std");
const contract = @import("zml-smi-2/contract");
const Metal = @import("zml-smi-2/drivers/metal");
const source = @import("zml-smi-2/source");

const energyOf = Metal.testing.energyOf;
const isPmpGroup = Metal.testing.isPmpGroup;
const isPmpCpuPower = Metal.testing.isPmpCpuPower;
const stateWatts = Metal.testing.stateWatts;
const milliwatts = Metal.testing.milliwatts;
const share = Metal.testing.share;

test "channel names" {
    try std.testing.expectEqual(.cpu, energyOf("CPU Energy"));
    try std.testing.expectEqual(.cpu, energyOf("DIE_0_CPU Energy"));
    try std.testing.expectEqual(.cpu_cluster, energyOf("PCPU Energy"));
    try std.testing.expectEqual(.gpu, energyOf("GPU Energy"));
    try std.testing.expectEqual(null, energyOf("GPU0"));
    try std.testing.expectEqual(.ane_block, energyOf("ANE0"));
    try std.testing.expectEqual(.ane, energyOf("ANE Energy"));
    try std.testing.expectEqual(null, energyOf("PACC_0"));

    try std.testing.expect(isPmpGroup("PMP") and isPmpGroup("PMP1") and !isPmpGroup("PMPX"));
    try std.testing.expect(isPmpCpuPower("PACC") and isPmpCpuPower("MACC1") and isPmpCpuPower("PACC0 SRAM"));
    try std.testing.expect(!isPmpCpuPower("AGX") and !isPmpCpuPower("AGX AFR") and !isPmpCpuPower("PACC SRAM2"));
    try std.testing.expectEqual(0.25, stateWatts(" 0.250W"));
    try std.testing.expectEqual(64, stateWatts("  64W"));

    // 1500 mJ in 500 ms.
    try std.testing.expectEqual(3000, milliwatts(1500, "mJ", .fromMilliseconds(500)));
    try std.testing.expectEqual(450, milliwatts(450_000_000, "nJ", .fromSeconds(1)));
}

test share {
    const second: std.Io.Duration = .fromSeconds(1);
    // Two processes with 300 ms and 100 ms of GPU time share a device at 80%.
    try std.testing.expectEqual(60, share(.{ .amount = 300_000_000, .elapsed = second }, 400_000_000, 80));
    try std.testing.expectEqual(20, share(.{ .amount = 100_000_000, .elapsed = second }, 400_000_000, 80));
    try std.testing.expectEqual(30, share(.{ .amount = 300_000_000, .elapsed = second }, 400_000_000, null));
}

test "a GPU of this Mac" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var metal = Metal.open(arena.allocator(), std.testing.allocator, io) catch
        return error.SkipZigTest;
    defer metal.close();

    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    var samples: [1]contract.Metrics = .{.{}};
    var reading: source.Reading = .{
        .samples = &samples,
        .allocator = std.testing.allocator,
    };
    defer reading.processes.deinit(std.testing.allocator);
    for (0..2) |round| {
        samples[0] = .{};
        reading.processes.clearRetainingCapacity();
        reading.host_power = .{};
        if (round > 0) try io.sleep(.fromMilliseconds(100), .awake);
        try metal.poll(scratch.allocator(), &reading);
    }
    const gpu = &samples[0];
    try std.testing.expect(gpu.get(.util_percent).? <= 100);
    try std.testing.expect(gpu.get(.mem_used_bytes).? < gpu.get(.mem_total_bytes).?);
    try std.testing.expect(gpu.get(.clock_graphics_mhz).? <= gpu.get(.clock_graphics_max_mhz).?);
    try std.testing.expect(gpu.get(.power_mw) != null and reading.host_power.cpu_power_mw != null);
    // WindowServer draws the screen with the GPU.
    try std.testing.expect(reading.processes.items.len > 0);
}
