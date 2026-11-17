//! Formats metric samples using the supplied display profile.
const std = @import("std");
const contract = @import("zml-smi-2/contract");
const data = @import("data.zig");
const image_cache = @import("image_cache.zig");
const spec = @import("spec.zig");
const utils = @import("lib/utils.zig");
const Gauge = @import("widgets/gauge.zig");
const MetricCard = @import("widgets/metric_card.zig");
const Chart = @import("widgets/chart.zig");

const OverviewDevice = struct { card: MetricCard, gauges: [2]Gauge };
const AllocError = std.mem.Allocator.Error;
const Metrics = contract.Metrics;
const Metric = data.Metric;

pub fn overviewDevice(
    arena: std.mem.Allocator,
    state: *const data.SystemState,
    index: u16,
) AllocError!OverviewDevice {
    const device = state.devices[index];
    const reading = &state.frame.samples[index];
    const style = device.profile;
    return .{
        .card = .{
            .title = device.title,
            .title_image = image_cache.get(device.logo),
            .title_right = if (device.remote) "remote" else null,
        },
        .gauges = .{
            gauge(reading, .util, style.util_label, try overviewSuffix(arena, style, reading)),
            gauge(reading, .memory, "MEM", try memoryInfo(arena, reading, "", "", true)),
        },
    };
}

fn gauge(sample: *const Metrics, kind: Metric, label: []const u8, suffix: []const u8) Gauge {
    return .{
        .label = label,
        .value = if (kind.read(sample)) |value| @intCast(@min(value, 100)) else null,
        .suffix = suffix,
    };
}

fn overviewSuffix(
    arena: std.mem.Allocator,
    style: *const spec.Profile,
    reading: *const Metrics,
) AllocError![]const u8 {
    if (style.environment == .hidden) return "";
    const temperature = try metric(arena, Metric.temperature.read(reading), "\u{00b0}C");
    const power = Metric.power.read(reading);
    if (style.environment == .tenths) return std.fmt.allocPrint(arena, "{s} | {s}", .{
        temperature, try watts(arena, power, "", " W"),
    });

    // Pad the power to the width of the limit to keep it aligned.
    const label = try metric(arena, if (power) |value| value / 1000 else null, "");
    const width = if (reading.get(.power_limit_mw)) |limit| std.fmt.count("{d}", .{limit / 1000}) else 0;
    return std.fmt.allocPrint(arena, "{[temperature]s} | {[power]s:>[width]} W", .{
        .temperature = temperature,
        .power = label,
        .width = width,
    });
}

const axes: std.EnumArray(Metric, struct { title: []const u8, unit: []const u8 }) = .init(.{
    .util = .{ .title = "Utilization", .unit = "%" },
    .memory = .{ .title = "Memory", .unit = "%" },
    .temperature = .{ .title = "Temperature", .unit = "\u{00b0}C" },
    .power = .{ .title = "Power", .unit = "W" },
});

/// Creates a Chart pre-filled with history data and standard defaults.
pub fn historyChart(
    arena: std.mem.Allocator,
    state: *const data.SystemState,
    index: u16,
    kind: Metric,
) AllocError!Chart {
    const profile = state.devices[index].profile;
    const reading = &state.frame.samples[index];
    const value = kind.read(reading);
    const axis = axes.get(kind);
    const history = try state.history(arena, index, kind);
    const minimum: u32 = if (kind == .temperature) 20 else 0;
    const limit = reading.get(.power_limit_mw);
    const maximum: u32 = if (kind == .power) maximum_blk: {
        var ceiling = limit orelse 0;
        if (ceiling == 0) {
            ceiling = @intCast(value orelse 0);
            for (history) |sample| ceiling = @max(ceiling, sample orelse 0);
        }
        const whole_watts = @divCeil(ceiling, 1000);
        break :maximum_blk std.math.lossyCast(u32, @max(2, whole_watts));
    } else 100;
    // Normalize power in milliwatts while labeling its axis in watts.
    const scale: u64 = if (kind == .power) 1000 else 1;
    const note = try detailLabel(arena, reading, profile, kind);
    return .{
        .title = if (kind == .util) profile.util_title else axis.title,
        .data = try utils.normalizeRange(arena, history, minimum, @as(u64, maximum) * scale),
        .value_label = if (kind == .power)
            try watts(arena, value, "", "W")
        else
            try metric(arena, value, axis.unit),
        .info_line = note orelse if (kind == .power)
            try watts(arena, if ((limit orelse 0) > 0) limit else null, "Limit ", "W")
        else
            null,
        .y_min = minimum,
        .y_max = maximum,
        .y_unit = axis.unit,
        .sample_interval_ms = if (state.frame.history) |history_data| history_data.interval_ms else 1,
    };
}

fn metric(arena: std.mem.Allocator, value: anytype, suffix: []const u8) AllocError![]const u8 {
    const number = value orelse return "N/A";
    return std.fmt.allocPrint(arena, "{d}{s}", .{ number, suffix });
}

fn memoryInfo(
    arena: std.mem.Allocator,
    reading: *const contract.Metrics,
    prefix: []const u8,
    suffix: []const u8,
    right_align: bool,
) std.mem.Allocator.Error![]const u8 {
    const used = try metric(arena, utils.mebibytes(reading.get(.mem_used_bytes)), "");
    const total = try metric(arena, utils.mebibytes(reading.get(.mem_total_bytes)), "");
    return std.fmt.allocPrint(arena, "{[prefix]s}{[used]s:>[width]} / {[total]s} MB{[suffix]s}", .{
        .width = if (right_align) total.len else 0,
        .prefix = prefix,
        .used = used,
        .total = total,
        .suffix = suffix,
    });
}

fn watts(
    arena: std.mem.Allocator,
    milliwatts: anytype,
    comptime prefix: []const u8,
    comptime suffix: []const u8,
) std.mem.Allocator.Error![]const u8 {
    const value = milliwatts orelse return prefix ++ "N/A";
    return std.fmt.allocPrint(arena, prefix ++ "{d}.{d}" ++ suffix, .{ value / 1000, value % 1000 / 100 });
}

pub fn detailLabel(
    arena: std.mem.Allocator,
    reading: *const Metrics,
    style: *const spec.Profile,
    kind: Metric,
) AllocError!?[]const u8 {
    switch (kind) {
        .memory => {
            const suffix = switch (style.memory_suffix) {
                .none => "",
                .literal => |text| text,
                .bus_width => |field| if (reading.get(field)) |width|
                    try std.fmt.allocPrint(arena, " ({d}-bit)", .{width})
                else
                    "",
            };
            return try memoryInfo(arena, reading, style.memory_prefix, suffix, false);
        },
        .util => {
            const notes = style.util_note orelse return null;
            const first = reading.get(notes[0].metric);
            const second = reading.get(notes[1].metric);
            if (first == null and second == null) return null;
            return try std.fmt.allocPrint(arena, "{s} {s}  {s} {s}", .{
                notes[0].label, try metric(arena, first, notes[0].suffix),
                notes[1].label, try metric(arena, second, notes[1].suffix),
            });
        },
        .temperature, .power => {
            const note = (if (kind == .power) style.power_note else style.temperature_note) orelse
                return null;
            const value = reading.get(note.metric);
            const label = if (kind == .power) try watts(arena, value, "", "W") else try metric(arena, value, note.suffix);
            return try std.fmt.allocPrint(arena, "{s} {s}", .{ note.label, label });
        },
    }
}

/// All returned card lines and formatted strings live in the frame arena.
pub fn cards(
    arena: std.mem.Allocator,
    sections: []const spec.Section,
    sample: *const Metrics,
    host: contract.HostData,
) AllocError![]const MetricCard {
    const result = try arena.alloc(MetricCard, sections.len);
    for (sections, result) |section, *card| {
        var lines = try std.ArrayList(MetricCard.LineEntry).initCapacity(arena, section.rows.len);
        for (section.rows) |row| {
            const value = try rowValue(arena, row, sample, host) orelse continue;
            lines.appendAssumeCapacity(.{ .label = row.label, .value = value });
        }
        card.* = .{ .title = section.title, .lines = lines.items };
    }
    return result;
}

fn rowValue(
    arena: std.mem.Allocator,
    row: spec.Row,
    sample: *const Metrics,
    host: contract.HostData,
) AllocError!?[]const u8 {
    return switch (row.value) {
        .clock_pair => |pair| try pairLabel(arena, " {d} / {d} MHz", sample.get(pair[0]), sample.get(pair[1])),
        .unit => |unit| try std.fmt.allocPrint(arena, " {d} {s}", .{
            sample.get(unit.metric) orelse return null,
            unit.suffix,
        }),
        .pcie_link => |pair| try pairLabel(arena, " Gen{d} x{d}", sample.get(pair[0]), sample.get(pair[1])),
        .traffic => |pair| traffic_blk: {
            const value = sample.get(pair[0]);
            if (value == null and sample.get(pair[1]) == null) break :traffic_blk null;
            break :traffic_blk try bandwidth(arena, value);
        },
        .watts => |field| try watts(arena, sample.get(field), " ", "W"),
        .host_watts => |field| switch (field) {
            inline else => |key| try watts(arena, @field(host, @tagName(key)), " ", "W"),
        },
        .named => |named| try code(arena, sample.get(named.metric), named.labels),
        .flags => |named| try flags(arena, sample.get(named.metric), named.labels),
        .memory => |field| if (utils.mebibytes(sample.get(field))) |memory|
            try std.fmt.allocPrint(arena, " {d} MB", .{memory})
        else
            " N/A",
    };
}

fn pairLabel(arena: std.mem.Allocator, comptime fmt: []const u8, left: ?u64, right: ?u64) AllocError![]const u8 {
    if (left == null and right == null) return " N/A";
    return std.fmt.allocPrint(arena, fmt, .{ left orelse 0, right orelse 0 });
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

fn code(arena: std.mem.Allocator, value: ?u64, labels: []const [:0]const u8) AllocError![]const u8 {
    const number = value orelse return " N/A";
    if (number < labels.len) return std.fmt.allocPrint(arena, " {s}", .{labels[@intCast(number)]});
    return std.fmt.allocPrint(arena, " {d}", .{number});
}

/// Unknown bits retain their positions in the displayed bitmask.
fn flags(arena: std.mem.Allocator, value: ?u64, labels: []const [:0]const u8) AllocError![]const u8 {
    const mask = value orelse return " N/A";
    if (mask == 0) return " none";
    var text: std.Io.Writer.Allocating = .init(arena);
    for (0..64) |bit| {
        if (mask >> @intCast(bit) & 1 == 0) continue;
        const separator = if (text.written().len == 0) " " else ", ";
        if (bit < labels.len) {
            text.writer.print("{s}{s}", .{ separator, labels[bit] }) catch return error.OutOfMemory;
        } else {
            text.writer.print("{s}bit {d}", .{ separator, bit }) catch return error.OutOfMemory;
        }
    }
    return text.written();
}
