const std = @import("std");
const sysfs = @import("../../utils/sysfs.zig");
const tpuinfo = @import("tpuinfo.zig");
const device_info = @import("../../info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const TpuInfo = device_info.TpuInfo;
const Collector = @import("../../collector.zig").Collector;
const Worker = @import("../../worker.zig").Worker;
const tpu_process = @import("process.zig");

const address = "localhost:8431";

pub const target: device_info.Target = .tpu;

pub fn start(collector: *Collector) !void {
    const chip = scanPciChips(collector.io) orelse return error.TpuUnavailable;
    const device_count = chip.chip_count * chip.devices_per_chip;
    var tpu_infos: std.ArrayList(*DeviceInfo) = .{};
    const dev_offset: u8 = @intCast(collector.device_infos.items.len);

    for (0..device_count) |_| {
        const chip_name = try collector.arena.dupe(u8, chip.name);
        const initial: TpuInfo = .{ .name = chip_name };
        const info = try collector.addDevice(.{ .tpu = .{ .values = .{ initial, initial } } });
        try tpu_infos.append(collector.arena, info);
    }

    try collector.worker.spawn(collector.io, pollAllDevices, .{ collector.io, collector.worker, tpu_infos.items });

    if (tpu_infos.items.len > 0) {
        const processes = try collector.createProcessList();
        try tpu_process.init(collector.worker, collector.io, collector.gpa, chip.devices_per_chip, tpu_infos.items, processes, dev_offset);
    }
}

const MetricKind = enum { int, double };

const metrics = .{
    .{ .field = "mem_used_bytes", .metric = "tpu.runtime.hbm.memory.usage.bytes", .kind = MetricKind.int },
    .{ .field = "mem_total_bytes", .metric = "tpu.runtime.hbm.memory.total.bytes", .kind = MetricKind.int },
    .{ .field = "util_percent", .metric = "tpu.runtime.tensorcore.dutycycle.percent", .kind = MetricKind.double },
};

fn pollAllDevices(io: std.Io, w: *const Worker, infos: []*DeviceInfo) void {
    w.pollLoop(io, struct {
        fn poll(devs: []*DeviceInfo) void {
            inline for (metrics) |m| {
                queryMetric(devs, m.field, m.metric, m.kind);
            }
        }
    }.poll, .{infos});
}

fn queryMetric(infos: []*DeviceInfo, comptime field: []const u8, metric_name: [:0]const u8, kind: MetricKind) void {
    var ids: [tpuinfo.max_devices]c_longlong = undefined;

    switch (kind) {
        .int => {
            var vals: [tpuinfo.max_devices]c_longlong = undefined;
            const n = tpuinfo.queryInt(address, metric_name, &ids, &vals) catch {
                for (infos) |info| {
                    const back = info.tpu.back();
                    back.* = info.tpu.front().*;
                    @field(back, field) = null;
                    info.tpu.swap();
                }
                return;
            };
            for (ids[0..n], vals[0..n]) |id, val| {
                if (id >= 0 and id < infos.len) {
                    const back = infos[@intCast(id)].tpu.back();
                    back.* = infos[@intCast(id)].tpu.front().*;
                    @field(back, field) = @intCast(val);
                    infos[@intCast(id)].tpu.swap();
                }
            } // Having not every value should in practice never happen
        },
        .double => {
            var vals: [tpuinfo.max_devices]f64 = undefined;
            const n = tpuinfo.queryDouble(address, metric_name, &ids, &vals) catch {
                for (infos) |info| {
                    const back = info.tpu.back();
                    back.* = info.tpu.front().*;
                    @field(back, field) = null;
                    info.tpu.swap();
                }
                return;
            };
            for (ids[0..n], vals[0..n]) |id, val| {
                if (id >= 0 and id < infos.len and std.math.isFinite(val)) {
                    const back = infos[@intCast(id)].tpu.back();
                    back.* = infos[@intCast(id)].tpu.front().*;
                    @field(back, field) = @intFromFloat(@round(val));
                    infos[@intCast(id)].tpu.swap();
                }
            } // Having not every value should in practice never happen
        },
    }
}

// -- Discover available chips
// We need to scan /sys/bus/pci/devices for Google TPU chips.
// From: https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/blob/c94f463dd3bf99eaecf4f5af46f87cc2a5ff1b22/tpu_info/tpu_info/device.py#L43-L92

const ChipInfo = struct {
    name: []const u8,
    devices_per_chip: u32,
    chip_count: u32,
};

const google_pci_vendor_id = "0x1ae0";
const pci_base = "/sys/bus/pci/devices";

const ChipEntry = struct {
    device_id: []const u8,
    subsystem_id: ?[]const u8 = null,
    name: []const u8,
    devices_per_chip: u32,
};

const chip_table = [_]ChipEntry{
    .{ .device_id = "0x0027", .subsystem_id = "0x004e", .name = "TPU v2", .devices_per_chip = 2 },
    .{ .device_id = "0x0027", .subsystem_id = "0x004f", .name = "TPU v3", .devices_per_chip = 2 },
    .{ .device_id = "0x005e", .name = "TPU v4", .devices_per_chip = 1 },
    .{ .device_id = "0x0063", .name = "TPU v5e", .devices_per_chip = 1 },
    .{ .device_id = "0x0062", .name = "TPU v5p", .devices_per_chip = 1 },
    .{ .device_id = "0x006f", .name = "TPU v6e", .devices_per_chip = 1 },
};

fn scanPciChips(io: std.Io) ?ChipInfo {
    var pci_dir = std.Io.Dir.openDir(.cwd(), io, pci_base, .{ .iterate = true }) catch return null;
    defer pci_dir.close(io);

    var it = pci_dir.iterate();
    var chip_count: u32 = 0;
    var result: ChipInfo = undefined;

    while (it.next(io) catch null) |entry| {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        var read_buf: [256]u8 = undefined;
        const vendor = readSysfs(io, &path_buf, &read_buf, entry.name, "vendor") catch continue;
        if (!std.mem.eql(u8, vendor, google_pci_vendor_id)) continue;

        var device_buf: [256]u8 = undefined;
        const device_raw = readSysfs(io, &path_buf, &device_buf, entry.name, "device") catch continue;
        var subsys_buf: [256]u8 = undefined;
        const subsystem_raw: ?[]const u8 = readSysfs(io, &path_buf, &subsys_buf, entry.name, "subsystem_device") catch null;

        for (chip_table) |chip| {
            const dev_match = std.mem.eql(u8, device_raw, chip.device_id);
            const sub_match = if (chip.subsystem_id) |expected_sub|
                if (subsystem_raw) |actual_sub| std.mem.eql(u8, actual_sub, expected_sub) else false
            else
                true;

            if (dev_match and sub_match) {
                result = .{
                    .name = chip.name,
                    .devices_per_chip = chip.devices_per_chip,
                    .chip_count = undefined,
                };
                chip_count += 1;
                break;
            }
        }
    }

    if (chip_count > 0) {
        result.chip_count = chip_count;
        return result;
    }

    return null;
}

fn readSysfs(io: std.Io, path_buf: *[std.Io.Dir.max_path_bytes]u8, read_buf: *[256]u8, slot: []const u8, file: []const u8) ![]const u8 {
    const path = std.fmt.bufPrint(path_buf, pci_base ++ "/{s}/{s}", .{ slot, file }) catch return error.Overflow;
    return sysfs.readString(io, path, read_buf);
}
