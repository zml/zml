//! The information unique to a platform; chart and page layout are shared.
const std = @import("std");
const smi = @import("zml-smi");
const utils = @import("../../lib/utils.zig");
const MetricCard = @import("../../widgets/metric_card.zig");
const common = @import("detail_common.zig");

/// All returned card lines and formatted strings live in the frame arena.
pub fn forSample(
    arena: std.mem.Allocator,
    sample: smi.Sample,
    host: smi.HostData,
) std.mem.Allocator.Error![]const MetricCard {
    const count: usize = switch (sample) {
        .cuda, .rocm, .oneapi, .metal => 2,
        .furiosa, .neuron => 1,
        .tpu => return &.{},
    };
    const result = try arena.alloc(MetricCard, count);
    switch (sample) {
        .cuda, .rocm, .oneapi => |gpu| {
            var clocks: std.ArrayList(MetricCard.LineEntry) = .empty;
            try clocks.append(arena, .{ .label = "Graphics ", .value = try clockPair(
                arena,
                gpu.clock_graphics_mhz,
                gpu.clock_graphics_max_mhz,
            ) });
            if (gpu.clock_sm_mhz) |value| try clocks.append(arena, .{
                .label = "SM       ",
                .value = try std.fmt.allocPrint(arena, " {d} MHz", .{value}),
            });
            if (gpu.clock_soc_mhz) |value| try clocks.append(arena, .{
                .label = "SOC      ",
                .value = try std.fmt.allocPrint(arena, " {d} MHz", .{value}),
            });
            try clocks.append(arena, .{ .label = "Memory   ", .value = try clockPair(
                arena,
                gpu.clock_mem_mhz,
                gpu.clock_mem_max_mhz,
            ) });
            result[0] = .{ .title = "Clocks", .lines = clocks.items };

            var pcie: std.ArrayList(MetricCard.LineEntry) = .empty;
            try pcie.append(arena, .{ .label = "Link     ", .value = if (gpu.pcie_link_gen == null and gpu.pcie_link_width == null) " N/A" else try std.fmt.allocPrint(arena, " Gen{d} x{d}", .{
                gpu.pcie_link_gen orelse 0, gpu.pcie_link_width orelse 0,
            }) });
            if (gpu.pcie_tx_kbps != null or gpu.pcie_rx_kbps != null) {
                try pcie.append(arena, .{
                    .label = "TX       ",
                    .value = try bandwidth(arena, gpu.pcie_tx_kbps),
                });
                try pcie.append(arena, .{
                    .label = "RX       ",
                    .value = try bandwidth(arena, gpu.pcie_rx_kbps),
                });
            }
            if (gpu.pcie_bandwidth_mbps) |value| try pcie.append(arena, .{
                .label = "BW       ",
                .value = try std.fmt.allocPrint(arena, " {d} Mb/s", .{value}),
            });
            result[1] = .{ .title = "PCIe", .lines = pcie.items };
        },
        .metal => |gpu| {
            result[0] = .{ .title = "Clocks", .lines = try arena.dupe(MetricCard.LineEntry, &.{.{
                .label = "Graphics ",
                .value = try clockPair(arena, gpu.clock_graphics_mhz, gpu.clock_graphics_max_mhz),
            }}) };
            const powers = .{
                .{ "GPU     ", gpu.power_mw },      .{ "ANE     ", gpu.ane_power_mw },
                .{ "CPU     ", host.cpu_power_mw }, .{ "System  ", host.system_power_mw },
            };
            const lines = try arena.alloc(MetricCard.LineEntry, powers.len);
            inline for (powers, lines) |power, *line| line.* = .{
                .label = power[0],
                .value = try std.fmt.allocPrint(arena, " {s}", .{try common.watts(arena, power[1])}),
            };
            result[1] = .{ .title = "Power", .lines = lines };
        },
        .furiosa => |chip| {
            result[0] = .{ .title = "State", .lines = try arena.dupe(MetricCard.LineEntry, &.{
                .{ .label = "Liveness ", .value = try code(arena, smi.Furiosa.Liveness, chip.liveness) },
                .{ .label = "Governor ", .value = try code(arena, smi.Furiosa.Governor, chip.governor) },
                .{ .label = "Throttle ", .value = try flags(arena, chip.throttle_reasons) },
            }) };
        },
        .neuron => |core| {
            const categories = .{
                .{ "Tensors       ", core.nc_tensors },
                .{ "Constants     ", core.nc_constants },
                .{ "Model Code    ", core.nc_model_code },
                .{ "Shared Scratch", core.nc_shared_scratchpad },
                .{ "Nonshared     ", core.nc_nonshared_scratchpad },
                .{ "Runtime       ", core.nc_runtime },
                .{ "Driver        ", core.nc_driver },
                .{ "DMA Rings     ", core.nc_dma_rings },
                .{ "Collectives   ", core.nc_collectives },
                .{ "Notifications ", core.nc_notifications },
                .{ "Uncategorized ", core.nc_uncategorized },
            };
            const lines = try arena.alloc(MetricCard.LineEntry, categories.len);
            inline for (categories, lines) |category, *line| line.* = .{
                .label = category[0],
                .value = if (category[1]) |value|
                    try std.fmt.allocPrint(arena, " {d} MB", .{utils.bytesToMb(value)})
                else
                    " N/A",
            };
            result[0] = .{ .title = "HBM Breakdown", .lines = lines };
        },
        .tpu => unreachable,
    }
    return result;
}

fn clockPair(arena: std.mem.Allocator, current: ?u64, maximum: ?u64) ![]const u8 {
    if (current == null and maximum == null) return " N/A";
    return std.fmt.allocPrint(arena, " {d} / {d} MHz", .{ current orelse 0, maximum orelse 0 });
}

fn bandwidth(arena: std.mem.Allocator, value: ?u64) ![]const u8 {
    const kbps = value orelse return " N/A";
    if (kbps < 1_000) return std.fmt.allocPrint(arena, " {d} KB/s", .{kbps});
    const divisor: f64 = if (kbps >= 1_000_000) 1_000_000 else 1_000;
    return std.fmt.allocPrint(arena, " {d:.1} {s}", .{
        @as(f64, @floatFromInt(kbps)) / divisor,
        if (kbps >= 1_000_000) "GB/s" else "MB/s",
    });
}

/// Preserve the raw number when firmware reports an unknown enum value.
fn code(arena: std.mem.Allocator, comptime Enum: type, value: ?u64) ![]const u8 {
    const number = value orelse return " N/A";
    if (std.enums.fromInt(Enum, number)) |known| return std.fmt.allocPrint(arena, " {t}", .{known});
    return std.fmt.allocPrint(arena, " {d}", .{number});
}

/// Unknown throttle bits retain their positions in the displayed bitmask.
fn flags(arena: std.mem.Allocator, value: ?u64) ![]const u8 {
    const mask = value orelse return " N/A";
    if (mask == 0) return " none";
    var text: std.Io.Writer.Allocating = .init(arena);
    for (0..64) |bit| {
        if (mask >> @intCast(bit) & 1 == 0) continue;
        const separator = if (text.written().len == 0) " " else ", ";
        if (std.enums.fromInt(smi.Furiosa.ThrottleReason, bit)) |flag| {
            text.writer.print("{s}{t}", .{ separator, flag }) catch return error.OutOfMemory;
        } else {
            text.writer.print("{s}bit {d}", .{ separator, bit }) catch return error.OutOfMemory;
        }
    }
    return text.written();
}
