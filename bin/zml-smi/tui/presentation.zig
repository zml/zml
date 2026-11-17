//! Platform labels, formatting, and detail-card content for the shared layouts.
const std = @import("std");
const vxfw = @import("vaxis").vxfw;
const model = @import("zml-smi/model");
const data = @import("data.zig");
const image_cache = @import("image_cache.zig");
const theme = @import("theme.zig");
const utils = @import("lib/utils.zig");
const Gauge = @import("widgets/gauge.zig");
const MetricCard = @import("widgets/metric_card.zig");
const common = @import("pages/detail/detail_common.zig");

const Profile = struct {
    device_label: []const u8 = "GPU",
    util_label: []const u8 = "GPU",
    util_title: []const u8 = "GPU Utilization",
    memory_prefix: []const u8 = "VRAM ",
    environment: enum { whole, decimal, hidden } = .whole,
};

fn profile(target: model.Target) Profile {
    return switch (target) {
        .cuda, .rocm, .oneapi => .{},
        .metal => .{ .memory_prefix = "In use ", .environment = .decimal },
        .furiosa => .{
            .device_label = "NPU",
            .util_label = "NPU",
            .util_title = "NPU Utilization",
            .memory_prefix = "DRAM ",
        },
        .neuron => .{
            .device_label = "NC",
            .util_label = "Core",
            .util_title = "Core Utilization",
            .memory_prefix = "HBM ",
            .environment = .hidden,
        },
        .tpu => .{
            .device_label = "TPU",
            .util_label = "Duty",
            .util_title = "Duty Cycle",
            .memory_prefix = "HBM ",
            .environment = .hidden,
        },
    };
}

const OverviewDevice = struct { card: MetricCard, gauges: [2]Gauge };

pub fn overviewDevice(arena: std.mem.Allocator, frame: model.Frame, index: u16) std.mem.Allocator.Error!OverviewDevice {
    const device = frame.devices[index];
    const reading = frame.samples[index].common();
    const style = profile(device.target);
    const total = try common.memoryLabel(arena, reading.mem_total_bytes);
    const used = try common.memoryLabel(arena, reading.mem_used_bytes);
    const pad = try utils.repeatStr(arena, " ", @intCast(total.len -| used.len));
    return .{
        .card = .{
            .title = try std.fmt.allocPrint(arena, "{s} {d}: {s}{s}", .{
                style.device_label, index, device.name orelse "Unknown", try utils.versions(arena, device, .in_title),
            }),
            .title_image = image_cache.global.get(device.target),
            .title_right = if (device.remote) "remote" else null,
        },
        .gauges = .{
            .{
                .label = style.util_label,
                .value = if (reading.util_percent) |value| @intCast(@min(value, 100)) else null,
                .suffix = try overviewSuffix(arena, style, reading),
            },
            .{
                .label = "MEM",
                .value = if (data.memoryPercent(reading)) |value| @intCast(value) else null,
                .suffix = try std.fmt.allocPrint(arena, "{s}{s} / {s} MB", .{ pad, used, total }),
            },
        },
    };
}

fn overviewSuffix(arena: std.mem.Allocator, style: Profile, reading: model.Common) std.mem.Allocator.Error![]const u8 {
    if (style.environment == .hidden) return "";
    const temperature = try common.metric(arena, reading.temperature, "\u{00b0}C");
    if (style.environment == .decimal) {
        const power = if (reading.power_mw) |milliwatts|
            try std.fmt.allocPrint(arena, "{d}.{d} W", .{ milliwatts / 1000, milliwatts % 1000 / 100 })
        else
            "N/A";
        return std.fmt.allocPrint(arena, "{s} | {s}", .{ temperature, power });
    }

    // Pad the power to the width of the limit to keep it aligned.
    const power = try common.metric(arena, if (reading.power_mw) |value| value / 1000 else null, "");
    const limit = if (reading.power_limit_mw) |value| try common.metric(arena, value / 1000, "") else power;
    const pad = try utils.repeatStr(arena, " ", @intCast(limit.len -| power.len));
    return std.fmt.allocPrint(arena, "{s} | {s}{s} W", .{ temperature, pad, power });
}

const DetailLabels = std.EnumArray(data.Metric, ?[]const u8);

const DetailView = struct {
    heading: vxfw.Text,
    reading: model.Common,
    util_title: []const u8,
    environment_charts: bool,
    labels: DetailLabels,
    cards: []const MetricCard,
};

pub fn detailView(arena: std.mem.Allocator, frame: model.Frame, index: u16) std.mem.Allocator.Error!DetailView {
    const device = frame.devices[index];
    const sample = frame.samples[index];
    const reading = sample.common();
    const style = profile(device.target);
    return .{
        .heading = .{
            .text = try std.fmt.allocPrint(arena, "Device {d}: {s}{s}", .{
                index, device.name orelse "Unknown", try utils.versions(arena, device, .all),
            }),
            .style = theme.header_style,
            .softwrap = false,
            .overflow = .clip,
        },
        .reading = reading,
        .util_title = style.util_title,
        .environment_charts = style.environment != .hidden,
        .labels = try detailLabels(arena, sample, reading, style),
        .cards = try cards(arena, sample, frame.host),
    };
}

fn detailLabels(
    arena: std.mem.Allocator,
    sample: model.Sample,
    reading: model.Common,
    style: Profile,
) std.mem.Allocator.Error!DetailLabels {
    var labels: DetailLabels = .initFill(null);
    var memory_suffix: []const u8 = "";
    switch (sample) {
        .cuda, .rocm, .oneapi => |gpu| {
            if (gpu.mem_bus_width) |width| memory_suffix = try std.fmt.allocPrint(arena, " ({d}-bit)", .{width});
            if (gpu.encoder_util_percent != null or gpu.decoder_util_percent != null) {
                labels.set(.util, try std.fmt.allocPrint(arena, "Enc {s}  Dec {s}", .{
                    try common.metric(arena, gpu.encoder_util_percent, "%"),
                    try common.metric(arena, gpu.decoder_util_percent, "%"),
                }));
            }
        },
        .metal => |gpu| {
            memory_suffix = " unified";
            labels.set(.power, try std.fmt.allocPrint(arena, "ANE {s}", .{try common.watts(arena, gpu.ane_power_mw)}));
        },
        .furiosa => |chip| labels.set(.temperature, try std.fmt.allocPrint(arena, "Ambient {s}", .{
            try common.metric(arena, chip.ambient_temperature, "\u{00b0}C"),
        })),
        .neuron, .tpu => {},
    }
    if (style.environment != .hidden and labels.get(.temperature) == null) {
        labels.set(.temperature, try std.fmt.allocPrint(arena, "Fan {s}", .{try common.metric(arena, reading.fan_speed_percent, "%")}));
    }
    labels.set(.memory, try common.memoryInfo(arena, reading, style.memory_prefix, memory_suffix));
    return labels;
}

const Line = MetricCard.LineEntry;

/// All returned card lines and formatted strings live in the frame arena.
fn cards(
    arena: std.mem.Allocator,
    sample: model.Sample,
    host: model.HostData,
) std.mem.Allocator.Error![]const MetricCard {
    return switch (sample) {
        .cuda, .rocm, .oneapi => |gpu| gpu_blk: {
            const traffic = gpu.pcie_tx_kbps != null or gpu.pcie_rx_kbps != null;
            const link = if (gpu.pcie_link_gen == null and gpu.pcie_link_width == null)
                " N/A"
            else
                try std.fmt.allocPrint(arena, " Gen{d} x{d}", .{ gpu.pcie_link_gen orelse 0, gpu.pcie_link_width orelse 0 });
            break :gpu_blk try arena.dupe(MetricCard, &.{
                try optionalLines(arena, "Clocks", &.{
                    .{ .label = "Graphics ", .value = try clockPair(arena, gpu.clock_graphics_mhz, gpu.clock_graphics_max_mhz) },
                    try unitLine(arena, "SM       ", gpu.clock_sm_mhz, "MHz"),
                    try unitLine(arena, "SOC      ", gpu.clock_soc_mhz, "MHz"),
                    .{ .label = "Memory   ", .value = try clockPair(arena, gpu.clock_mem_mhz, gpu.clock_mem_max_mhz) },
                }),
                try optionalLines(arena, "PCIe", &.{
                    .{ .label = "Link     ", .value = link },
                    if (traffic) .{ .label = "TX       ", .value = try bandwidth(arena, gpu.pcie_tx_kbps) } else null,
                    if (traffic) .{ .label = "RX       ", .value = try bandwidth(arena, gpu.pcie_rx_kbps) } else null,
                    try unitLine(arena, "BW       ", gpu.pcie_bandwidth_mbps, "Mb/s"),
                }),
            });
        },
        .metal => |gpu| metal_blk: {
            const powers = .{
                .{ "GPU     ", gpu.power_mw },      .{ "ANE     ", gpu.ane_power_mw },
                .{ "CPU     ", host.cpu_power_mw }, .{ "System  ", host.system_power_mw },
            };
            const lines = try arena.alloc(Line, powers.len);
            inline for (powers, lines) |power, *line| line.* = .{
                .label = power[0],
                .value = try std.fmt.allocPrint(arena, " {s}", .{try common.watts(arena, power[1])}),
            };
            break :metal_blk try arena.dupe(MetricCard, &.{
                .{ .title = "Clocks", .lines = try arena.dupe(Line, &.{
                    .{ .label = "Graphics ", .value = try clockPair(arena, gpu.clock_graphics_mhz, gpu.clock_graphics_max_mhz) },
                }) },
                .{ .title = "Power", .lines = lines },
            });
        },
        .furiosa => |chip| try arena.dupe(MetricCard, &.{.{
            .title = "State",
            .lines = try arena.dupe(Line, &.{
                .{ .label = "Liveness ", .value = try code(arena, model.Furiosa.Liveness, chip.liveness) },
                .{ .label = "Governor ", .value = try code(arena, model.Furiosa.Governor, chip.governor) },
                .{ .label = "Throttle ", .value = try flags(arena, chip.throttle_reasons) },
            }),
        }}),
        .neuron => |core| neuron_blk: {
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
            const lines = try arena.alloc(Line, categories.len);
            inline for (categories, lines) |category, *line| line.* = .{
                .label = category[0],
                .value = if (category[1]) |value|
                    try std.fmt.allocPrint(arena, " {d} MB", .{utils.bytesToMb(value)})
                else
                    " N/A",
            };
            break :neuron_blk try arena.dupe(MetricCard, &.{.{ .title = "HBM Breakdown", .lines = lines }});
        },
        .tpu => &.{},
    };
}

fn optionalLines(arena: std.mem.Allocator, title: []const u8, candidates: []const ?Line) std.mem.Allocator.Error!MetricCard {
    const lines = try arena.alloc(Line, candidates.len);
    var count: usize = 0;
    for (candidates) |candidate| {
        lines[count] = candidate orelse continue;
        count += 1;
    }
    return .{ .title = title, .lines = lines[0..count] };
}

fn unitLine(arena: std.mem.Allocator, label: []const u8, value: ?u64, unit: []const u8) std.mem.Allocator.Error!?Line {
    return .{ .label = label, .value = try std.fmt.allocPrint(arena, " {d} {s}", .{ value orelse return null, unit }) };
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
        if (std.enums.fromInt(model.Furiosa.ThrottleReason, bit)) |flag| {
            text.writer.print("{s}{t}", .{ separator, flag }) catch return error.OutOfMemory;
        } else {
            text.writer.print("{s}bit {d}", .{ separator, bit }) catch return error.OutOfMemory;
        }
    }
    return text.written();
}
