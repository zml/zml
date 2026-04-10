const std = @import("std");
const sysfs = @import("zml-smi/sysfs");
const tpuinfo = @import("tpuinfo.zig");
const device_info = @import("zml-smi/info").device_info;
const DeviceInfo = device_info.DeviceInfo;
const TpuInfo = device_info.TpuInfo;
const Collector = @import("zml-smi/collector").Collector;
const tpu_process = @import("process.zig");

const address = "localhost:8431";

pub fn start(collector: *Collector) !void {
    const chip = scanPciChips(collector.arena, collector.io) orelse return error.TpuUnavailable;
    const device_count = chip.chip_count * chip.devices_per_chip;
    var tpu_infos: std.ArrayList(*DeviceInfo) = .{};
    const dev_offset: u16 = @intCast(collector.device_infos.items.len);

    for (0..device_count) |_| {
        const chip_name = try collector.arena.dupe(u8, chip.name);
        const initial: TpuInfo = .{ .name = chip_name };
        const info = try collector.addDevice(.{ .tpu = .{ .values = .{ initial, initial } } });
        try tpu_infos.append(collector.arena, info);
    }

    try collector.spawnPoll(pollAllDevices, .{tpu_infos.items});

    if (tpu_infos.items.len > 0) {
        const processes = try collector.createProcessList();
        try tpu_process.init(collector, chip.devices_per_chip, tpu_infos.items, processes, dev_offset);
    }
}

const MetricKind = enum { int, double };

fn pollAllDevices(devs: []*DeviceInfo) void {
    var mem_used: [tpuinfo.max_devices]?u64 = undefined;
    var mem_total: [tpuinfo.max_devices]?u64 = undefined;
    var util: [tpuinfo.max_devices]?u64 = undefined;

    for (devs, 0..) |info, i| {
        const front = info.tpu.front();
        mem_used[i] = front.mem_used_bytes;
        mem_total[i] = front.mem_total_bytes;
        util[i] = front.util_percent;
    }

    queryRaw("tpu.runtime.hbm.memory.usage.bytes", .int, mem_used[0..devs.len]);
    queryRaw("tpu.runtime.hbm.memory.total.bytes", .int, mem_total[0..devs.len]);
    queryRaw("tpu.runtime.tensorcore.dutycycle.percent", .double, util[0..devs.len]);

    for (devs, 0..) |info, i| {
        const back = info.tpu.back();
        back.* = info.tpu.front().*;
        back.mem_used_bytes = mem_used[i];
        back.mem_total_bytes = mem_total[i];
        back.util_percent = util[i];
        info.tpu.swap();
    }
}

fn queryRaw(metric_name: [:0]const u8, comptime kind: MetricKind, out: []?u64) void {
    var ids: [tpuinfo.max_devices]c_longlong = undefined;
    switch (kind) {
        .int => {
            var vals: [tpuinfo.max_devices]c_longlong = undefined;
            const n = tpuinfo.queryInt(address, metric_name, &ids, &vals) catch {
                @memset(out, null);
                return;
            };

            for (ids[0..n], vals[0..n]) |id, val| {
                if (id >= 0 and id < out.len) out[@intCast(id)] = @intCast(val);
            }
        },
        .double => {
            var vals: [tpuinfo.max_devices]f64 = undefined;
            const n = tpuinfo.queryDouble(address, metric_name, &ids, &vals) catch {
                @memset(out, null);
                return;
            };

            for (ids[0..n], vals[0..n]) |id, val| {
                if (id >= 0 and id < out.len and std.math.isFinite(val))
                    out[@intCast(id)] = @intFromFloat(@round(val));
            }
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
    .{ .device_id = "0x0076", .name = "TPU7x", .devices_per_chip = 2 },
};

fn scanPciChips(allocator: std.mem.Allocator, io: std.Io) ?ChipInfo {
    var pci_dir = std.Io.Dir.openDir(.cwd(), io, pci_base, .{ .iterate = true }) catch return null;
    defer pci_dir.close(io);

    var it = pci_dir.iterate();
    var chip_count: u32 = 0;
    var result: ChipInfo = undefined;

    while (it.next(io) catch null) |entry| {
        const vendor = readSysfs(allocator, io, entry.name, "vendor") catch continue;
        if (!std.mem.eql(u8, vendor, google_pci_vendor_id)) {
            continue;
        }

        const device_raw = readSysfs(allocator, io, entry.name, "device") catch continue;
        const subsystem_raw: ?[]const u8 = readSysfs(allocator, io, entry.name, "subsystem_device") catch null;

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

fn readSysfs(allocator: std.mem.Allocator, io: std.Io, slot: []const u8, file: []const u8) ![]const u8 {
    const path = try std.fmt.allocPrint(allocator, pci_base ++ "/{s}/{s}", .{ slot, file });
    return sysfs.readString(allocator, io, path);
}
