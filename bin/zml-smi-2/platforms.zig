//! Application composition for native readers and device presentation.
const builtin = @import("builtin");
const std = @import("std");
const contract = @import("zml-smi-2/contract");
const spec = @import("zml-smi-2/tui").spec;

const Logo = enum { logo, nvidia, amd, intel, neuron, tpu };

/// Slot zero is the ZML logo used in the overview header.
pub const logos: []const []const u8 = &logo_pngs;

const logo_pngs = pngs_blk: {
    var pngs: [std.enums.values(Logo).len][]const u8 = undefined;
    for (std.enums.values(Logo)) |logo| {
        pngs[@backingInt(logo)] = @embedFile("assets/" ++ @tagName(logo) ++ ".png");
    }
    break :pngs_blk pngs;
};

const graphics_clock = row("Graphics ", .{ .clock_pair = .{
    .clock_graphics_mhz, .clock_graphics_max_mhz,
} });
const fan: spec.Note = .{ .label = "Fan", .metric = .fan_speed_percent, .suffix = "%" };

const gpu: spec.Profile = .{
    .util_label = "GPU",
    .util_title = "GPU Utilization",
    .memory_prefix = "VRAM ",
    .memory_suffix = .{ .bus_width = .mem_bus_width },
    .util_note = .{
        .{ .label = "Enc", .metric = .encoder_util_percent, .suffix = "%" },
        .{ .label = "Dec", .metric = .decoder_util_percent, .suffix = "%" },
    },
    .temperature_note = fan,
    .sections = &.{
        .{ .title = "Clocks", .rows = &.{
            graphics_clock,
            row("SM       ", .{ .unit = .{ .metric = .clock_sm_mhz, .suffix = "MHz" } }),
            row("SOC      ", .{ .unit = .{ .metric = .clock_soc_mhz, .suffix = "MHz" } }),
            row("Memory   ", .{ .clock_pair = .{ .clock_mem_mhz, .clock_mem_max_mhz } }),
        } },
        .{ .title = "PCIe", .rows = &.{
            row("Link     ", .{ .pcie_link = .{ .pcie_link_gen, .pcie_link_width } }),
            row("TX       ", .{ .traffic = .{ .pcie_tx_kbps, .pcie_rx_kbps } }),
            row("RX       ", .{ .traffic = .{ .pcie_rx_kbps, .pcie_tx_kbps } }),
            row("BW       ", .{ .unit = .{ .metric = .pcie_bandwidth_mbps, .suffix = "Mb/s" } }),
        } },
    },
};

const Entry = struct {
    profile: *const spec.Profile = &gpu,
    title_label: []const u8 = "GPU",
    logo: ?Logo = null,
};

const Platform = struct { display: Entry, driver: type };

/// Portable lookups supply display metadata without resolving any native driver.
pub fn lookup(comptime target: contract.Target, comptime mode: enum { native, portable }) Platform {
    const linux = mode == .native and builtin.os.tag == .linux;
    const linux_nvml = linux and (builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .aarch64);
    const linux_x86 = linux and builtin.cpu.arch == .x86_64;
    const mac_arm = mode == .native and builtin.os.tag == .macos and builtin.cpu.arch == .aarch64;
    const display: Entry, const backend = switch (target) {
        .cuda => .{ .{ .logo = .nvidia }, if (linux_nvml) @import("zml-smi-2/drivers/nvml") },
        .rocm => .{ .{ .logo = .amd }, if (linux_x86) @import("zml-smi-2/drivers/amdsmi") },
        .oneapi => .{ .{ .logo = .intel }, if (linux_x86) @import("zml-smi-2/drivers/oneapi") },
        .neuron => .{ .{ .title_label = "NC", .logo = .neuron, .profile = &.{
            .util_label = "Core",
            .util_title = "Core Utilization",
            .memory_prefix = "HBM ",
            .environment = .hidden,
            .sections = &.{.{ .title = "HBM Breakdown", .rows = &.{
                row("Tensors       ", .{ .memory = .nc_tensors }),
                row("Constants     ", .{ .memory = .nc_constants }),
                row("Model Code    ", .{ .memory = .nc_model_code }),
                row("Shared Scratch", .{ .memory = .nc_shared_scratchpad }),
                row("Nonshared     ", .{ .memory = .nc_nonshared_scratchpad }),
                row("Runtime       ", .{ .memory = .nc_runtime }),
                row("Driver        ", .{ .memory = .nc_driver }),
                row("DMA Rings     ", .{ .memory = .nc_dma_rings }),
                row("Collectives   ", .{ .memory = .nc_collectives }),
                row("Notifications ", .{ .memory = .nc_notifications }),
                row("Uncategorized ", .{ .memory = .nc_uncategorized }),
            } }},
        } }, if (linux_x86) @import("zml-smi-2/drivers/neuron") },
        .tpu => .{ .{ .title_label = "TPU", .logo = .tpu, .profile = &.{
            .util_label = "Duty",
            .util_title = "Duty Cycle",
            .memory_prefix = "HBM ",
            .environment = .hidden,
        } }, if (linux_x86) @import("zml-smi-2/drivers/tpu") },
        .furiosa => .{ .{ .title_label = "NPU", .profile = &.{
            .util_label = "NPU",
            .util_title = "NPU Utilization",
            .memory_prefix = "DRAM ",
            .temperature_note = .{
                .label = "Ambient",
                .metric = .ambient_temperature,
                .suffix = "\u{00b0}C",
            },
            .sections = &.{.{ .title = "State", .rows = &.{
                row("Liveness ", .{ .named = named(.liveness, contract.Liveness) }),
                row("Governor ", .{ .named = named(.governor, contract.Governor) }),
                row("Throttle ", .{ .flags = named(.throttle_reasons, contract.ThrottleReason) }),
            } }},
        } }, if (linux_x86) @import("zml-smi-2/drivers/furiosa") },
        .metal => .{ .{ .profile = &.{
            .util_label = "GPU",
            .util_title = "GPU Utilization",
            .memory_prefix = "In use ",
            .memory_suffix = .{ .literal = " unified" },
            .environment = .tenths,
            .temperature_note = fan,
            .power_note = .{ .label = "ANE", .metric = .ane_power_mw },
            .sections = &.{
                .{ .title = "Clocks", .rows = &.{graphics_clock} },
                .{ .title = "Power", .rows = &.{
                    row("GPU     ", .{ .watts = .power_mw }),
                    row("ANE     ", .{ .watts = .ane_power_mw }),
                    row("CPU     ", .{ .host_watts = .cpu_power_mw }),
                    row("System  ", .{ .host_watts = .system_power_mw }),
                } },
            },
        } }, if (mac_arm) @import("zml-smi-2/drivers/metal") },
    };
    return .{ .display = display, .driver = if (@TypeOf(backend) == type) backend else void };
}

fn row(label: []const u8, value: spec.Row.Value) spec.Row {
    return .{ .label = label, .value = value };
}

fn named(metric: contract.Metric, comptime Tag: type) spec.Named {
    comptime for (std.enums.values(Tag), 0..) |tag, index| {
        if (@backingInt(tag) != index) @compileError("expected dense enum: " ++ @typeName(Tag));
    };
    return .{ .metric = metric, .labels = @typeInfo(Tag).@"enum".field_names };
}

const DetailLabel = struct { label: []const u8, in_title: bool = false };
const detail_labels = std.StaticStringMap(DetailLabel).initComptime(.{
    .{ "cuda_version", DetailLabel{ .label = "CUDA", .in_title = true } },
    .{ "firmware_version", DetailLabel{ .label = "Firmware" } },
    .{ "serial", DetailLabel{ .label = "Serial" } },
});

/// Builds the terminal's device views and titles once in run storage.
pub fn devices(
    arena: std.mem.Allocator,
    inventory: []const contract.Device,
) std.mem.Allocator.Error![]const spec.DeviceView {
    const views = try arena.alloc(spec.DeviceView, inventory.len);
    for (inventory, views, 0..) |device, *view, index| {
        const entry = switch (device.target) {
            inline else => |target| lookup(target, .portable).display,
        };
        view.* = .{
            .title = title(arena, device, index, entry.title_label, .overview) catch
                return error.OutOfMemory,
            .heading = title(arena, device, index, "Device", .detail) catch return error.OutOfMemory,
            .remote = device.remote,
            .profile = entry.profile,
            .logo = if (entry.logo) |logo| @backingInt(logo) else null,
        };
    }
    return views;
}

fn title(
    arena: std.mem.Allocator,
    device: contract.Device,
    index: usize,
    label: []const u8,
    comptime page: enum { overview, detail },
) std.Io.Writer.Error![]const u8 {
    var text: std.Io.Writer.Allocating = .init(arena);
    errdefer text.deinit();
    const writer = &text.writer;
    try writer.print("{s} {d}: {s}", .{ label, index, device.name orelse "Unknown" });
    const name_end = text.written().len;
    if (device.driver_version) |driver| {
        if (driver.len != 0) try writer.print(" ({s}", .{driver});
    }
    for (device.details) |attribute| {
        const display = detail_labels.get(attribute.key) orelse DetailLabel{ .label = attribute.key };
        if (page == .overview and !display.in_title) continue;
        const separator = if (text.written().len == name_end) " (" else ", ";
        try writer.print("{s}{s} {s}", .{ separator, display.label, attribute.value });
    }
    if (text.written().len != name_end) try writer.writeByte(')');
    return text.written();
}

pub fn available(comptime Backend: type, io: std.Io) bool {
    if (!@hasDecl(Backend, "detection_paths")) return true;
    for (Backend.detection_paths) |path| {
        std.Io.Dir.cwd().access(io, path, .{}) catch continue;
        return true;
    }
    return false;
}
