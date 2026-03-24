const std = @import("std");
const DeviceInfo = @import("device_info.zig").DeviceInfo;
const HostInfo = @import("host_info.zig").HostInfo;
const Target = @import("main.zig").Target;

fn strSlice(buf: *const [256]u8) []const u8 {
    return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(buf)), 0);
}

fn drawHost(writer: *std.Io.Writer, host: *const HostInfo) !void {
    // Host / OS
    try writer.writeAll("Host:   ");
    if (host.hostname) |*buf| try writer.print("{s}", .{strSlice(buf)}) else try writer.writeAll("unknown");
    if (host.kernel) |*buf| try writer.print(" | Linux {s}", .{strSlice(buf)});
    try writer.writeAll("\n");

    // CPU
    if (host.cpu_name) |*buf| {
        try writer.print("CPU:    {s}", .{strSlice(buf)});
        if (host.cpu_cores) |cores| try writer.print(" ({d} cores)", .{cores});
        try writer.writeAll("\n");
    }

    // RAM
    if (host.mem_total_kib) |total_kib| {
        try writer.writeAll("RAM:    ");
        const total_gib = @as(f64, @floatFromInt(total_kib)) / (1024.0 * 1024.0);
        if (host.mem_available_kib) |avail_kib| {
            const used_kib = total_kib - avail_kib;
            const used_gib = @as(f64, @floatFromInt(used_kib)) / (1024.0 * 1024.0);
            try writer.print("{d:.1} / {d:.1} GiB", .{ used_gib, total_gib });
        } else {
            try writer.print("{d:.1} GiB total", .{total_gib});
        }
        try writer.writeAll("\n");
    }

    // Load / Temp / Uptime
    var has_line = false;
    if (host.load_avg) |*buf| {
        try writer.print("Load:   {s}", .{strSlice(buf)});
        has_line = true;
    }
    if (host.cpu_temp) |temp| {
        if (has_line) try writer.writeAll("  |  ") else try writer.writeAll("Temp:   ");
        try writer.print("{d}°C", .{temp});
        has_line = true;
    }
    if (host.uptime_seconds) |secs| {
        if (has_line) try writer.writeAll("  |  ") else try writer.writeAll("Up:     ");
        const days = secs / 86400;
        const hours = (secs % 86400) / 3600;
        const mins = (secs % 3600) / 60;
        if (days > 0) try writer.print("{d}d ", .{days});
        if (days > 0 or hours > 0) try writer.print("{d}h ", .{hours});
        try writer.print("{d}m", .{mins});
        has_line = true;
    }
    if (has_line) try writer.writeAll("\n");
}

const nc_mem_fields = .{
    .{ "nc_tensors", "tensors" },
    .{ "nc_constants", "constants" },
    .{ "nc_model_code", "model code" },
    .{ "nc_shared_scratchpad", "shared scratchpad" },
    .{ "nc_nonshared_scratchpad", "nonshared scratchpad" },
    .{ "nc_runtime", "runtime" },
    .{ "nc_driver", "driver" },
    .{ "nc_dma_rings", "dma rings" },
    .{ "nc_collectives", "collectives" },
    .{ "nc_notifications", "notifications" },
    .{ "nc_uncategorized", "uncategorized" },
};

fn drawNeuronMem(writer: *std.Io.Writer, info: DeviceInfo) !void {
    inline for (nc_mem_fields) |entry| {
        const val = @field(info, entry[0]) orelse 0;
        const mib = @as(f64, @floatFromInt(val)) / (1024.0 * 1024.0);
        try writer.print("    {s}: {d:.1} MiB\n", .{ entry[1], mib });
    }
}

pub fn draw(writer: *std.Io.Writer, target: ?Target, host: *const HostInfo, infos: []const DeviceInfo) !void {
    try writer.writeAll("\x1b[2J\x1b[H");
    try drawHost(writer, host);
    if (target) |t| {
        try writer.print("\nPlatform: {s}\n", .{@tagName(t)});
    } else if (infos.len == 0) {
        try writer.writeAll("\nNo accelerator platform found.\n");
        return;
    }
    const is_neuron = if (target) |t| t == .neuron else false;
    const is_tpu = if (target) |t| t == .tpu else false;
    for (infos, 0..) |info, i| {
        const name: []const u8 = if (info.name) |*buf| strSlice(buf) else "unknown";
        const label = if (is_neuron) "NeuronCore" else if (is_tpu) "TPU" else "GPU";
        try writer.print("\n\x1b[1m{s} {d}: {s}\x1b[0m\n", .{ label, i, name });

        // Power (not available on Neuron/TPU)
        if (info.power_mw != null or !(is_neuron or is_tpu)) {
            try writer.writeAll("  Power:  ");
            if (info.power_mw) |mw| {
                try writer.print("{d:.1}W", .{@as(f64, @floatFromInt(mw)) / 1000.0});
                if (info.power_limit_mw) |limit| {
                    try writer.print(" / {d:.1}W", .{@as(f64, @floatFromInt(limit)) / 1000.0});
                }
            } else {
                try writer.writeAll("N/A");
            }
            if (info.total_energy_mj) |mj| {
                try writer.print("  (total: {d:.1} kJ)", .{@as(f64, @floatFromInt(mj)) / 1_000_000.0});
            }
            try writer.writeAll("\n");
        }

        // Thermal (not available on Neuron/TPU)
        if (info.temperature != null or !(is_neuron or is_tpu)) {
            try writer.writeAll("  Temp:   ");
            if (info.temperature) |t| {
                try writer.print("{d}°C", .{t});
            } else {
                try writer.writeAll("N/A");
            }
            if (info.fan_speed_percent) |fan| {
                try writer.print("  Fan: {d}%", .{fan});
            }
            try writer.writeAll("\n");
        }

        // Utilization
        try writer.writeAll("  Util:   ");
        if (info.gpu_util_percent) |gpu| {
            try writer.print("{d}%", .{gpu});
        } else {
            try writer.writeAll("N/A");
        }
        if (info.mem_util_percent) |mem| {
            try writer.print("  Mem {d}%", .{mem});
        }
        if (info.encoder_util_percent) |enc| {
            if (enc > 0) try writer.print("  Enc {d}%", .{enc});
        }
        if (info.decoder_util_percent) |dec| {
            if (dec > 0) try writer.print("  Dec {d}%", .{dec});
        }
        try writer.writeAll("\n");

        // Clocks (not available on Neuron)
        if (info.clock_graphics_mhz != null or info.clock_mem_mhz != null) {
            try writer.writeAll("  Clocks: ");
            if (info.clock_graphics_mhz) |gfx| {
                try writer.print("GFX {d}", .{gfx});
                if (info.clock_graphics_max_mhz) |max| {
                    try writer.print("/{d}", .{max});
                }
                try writer.writeAll(" MHz");
            }
            if (info.clock_sm_mhz) |sm| {
                try writer.print("  SM {d} MHz", .{sm});
            }
            if (info.clock_mem_mhz) |mem| {
                try writer.print("  Mem {d}", .{mem});
                if (info.clock_mem_max_mhz) |max| {
                    try writer.print("/{d}", .{max});
                }
                try writer.writeAll(" MHz");
            }
            if (info.pstate) |p| {
                try writer.print("  P{d}", .{p});
            }
            try writer.writeAll("\n");
        }

        // Memory
        try writer.writeAll("  Mem:    ");
        if (info.mem_used_bytes) |used| {
            const used_mb = @as(f64, @floatFromInt(used)) / (1024.0 * 1024.0);
            if (info.mem_total_bytes) |total| {
                const total_mb = @as(f64, @floatFromInt(total)) / (1024.0 * 1024.0);
                try writer.print("{d:.0} / {d:.0} MiB", .{ used_mb, total_mb });
            } else {
                try writer.print("{d:.0} MiB used", .{used_mb});
            }
            if (info.mem_free_bytes) |free| {
                const free_mb = @as(f64, @floatFromInt(free)) / (1024.0 * 1024.0);
                try writer.print("  ({d:.0} MiB free)", .{free_mb});
            }
            if (info.mem_bus_width) |bw| {
                try writer.print("  {d}-bit", .{bw});
            }
        } else {
            try writer.writeAll("N/A");
        }
        try writer.writeAll("\n");

        // Neuron HBM breakdown
        if (is_neuron) {
            try drawNeuronMem(writer, info);
        }

        // PCIe (not available on Neuron)
        if (info.pcie_tx_kbps != null or info.pcie_rx_kbps != null or info.pcie_link_gen != null) {
            try writer.writeAll("  PCIe:   ");
            if (info.pcie_link_gen) |gen| {
                try writer.print("Gen{d}", .{gen});
                if (info.pcie_link_width) |w| {
                    try writer.print("x{d}", .{w});
                }
                try writer.writeAll("  ");
            }
            if (info.pcie_tx_kbps) |tx| {
                try writer.print("TX {d} KB/s", .{tx});
            }
            if (info.pcie_rx_kbps) |rx| {
                try writer.print("  RX {d} KB/s", .{rx});
            }
            try writer.writeAll("\n");
        }
    }
}
