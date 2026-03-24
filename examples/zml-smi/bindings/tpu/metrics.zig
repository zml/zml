const std = @import("std");
const sysfs = @import("../../sysfs.zig");
const tpuinfo = @import("tpuinfo.zig");
const DeviceInfo = @import("../../device_info.zig").DeviceInfo;
const worker = @import("../../worker.zig");

const address = "localhost:8431";

const ChipInfo = struct {
    name: [256]u8,
    devices_per_chip: u32,
    chip_count: u32,
};

const google_pci_vendor_id = "0x1ae0";
const pci_base = "/sys/bus/pci/devices";

const chip_table = [_]struct { device_id: []const u8, subsystem_id: ?[]const u8, name: []const u8, devices_per_chip: u32 }{
    .{ .device_id = "0x0027", .subsystem_id = "0x004e", .name = "TPU v2", .devices_per_chip = 2 },
    .{ .device_id = "0x0027", .subsystem_id = "0x004f", .name = "TPU v3", .devices_per_chip = 2 },
    .{ .device_id = "0x005e", .subsystem_id = null, .name = "TPU v4", .devices_per_chip = 1 },
    .{ .device_id = "0x0063", .subsystem_id = null, .name = "TPU v5e", .devices_per_chip = 1 },
    .{ .device_id = "0x0062", .subsystem_id = null, .name = "TPU v5p", .devices_per_chip = 1 },
    .{ .device_id = "0x006f", .subsystem_id = null, .name = "TPU v6e", .devices_per_chip = 1 },
};

fn toName(name: []const u8) [256]u8 {
    var buf: [256]u8 = .{0} ** 256;
    const len = @min(name.len, 255);
    @memcpy(buf[0..len], name[0..len]);
    return buf;
}

fn readSysfs(io: std.Io, path_buf: *[std.Io.Dir.max_path_bytes]u8, slot: []const u8, file: []const u8) ![256]u8 {
    const path = std.fmt.bufPrint(path_buf, pci_base ++ "/{s}/{s}", .{ slot, file }) catch return error.Overflow;
    return sysfs.readString(io, path);
}

fn sysfsEql(raw: [256]u8, expected: []const u8) bool {
    return std.mem.eql(u8, std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&raw)), 0), expected);
}

/// Scan /sys/bus/pci/devices for Google TPU chips.
fn scanPciChips(io: std.Io) ?ChipInfo {
    var pci_dir = std.Io.Dir.openDir(.cwd(), io, pci_base, .{ .iterate = true }) catch return null;
    defer pci_dir.close(io);

    var it = pci_dir.iterate();
    var chip_count: u32 = 0;
    var result: ChipInfo = undefined;

    while (it.next(io) catch null) |entry| {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const vendor = readSysfs(io, &path_buf, entry.name, "vendor") catch continue;
        if (!sysfsEql(vendor, google_pci_vendor_id)) continue;

        const device_raw = readSysfs(io, &path_buf, entry.name, "device") catch continue;
        const subsystem_raw = readSysfs(io, &path_buf, entry.name, "subsystem_device") catch null;

        for (chip_table) |chip| {
            const dev_match = sysfsEql(device_raw, chip.device_id);
            const sub_match = if (chip.subsystem_id) |expected_sub|
                if (subsystem_raw) |actual_sub| sysfsEql(actual_sub, expected_sub) else false
            else
                true;

            if (dev_match and sub_match) {
                result = .{
                    .name = toName(chip.name),
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

pub fn init(io: std.Io, allocator: std.mem.Allocator, device_infos: *std.ArrayList(DeviceInfo), signal: *worker.Signal) !void {
    const chip = scanPciChips(io) orelse return;
    const device_count = chip.chip_count * chip.devices_per_chip;

    for (0..device_count) |_| {
        const info = try device_infos.addOne(allocator);
        info.* = .{};
        info.name = chip.name;
    }

    inline for (batch_metrics) |metric| {
        try worker.spawnBatchWorker(io, device_infos.items, metric.query, signal);
    }
}

fn clearField(infos: []DeviceInfo, comptime field: []const u8) void {
    for (infos) |*info| {
        @field(info, field) = null;
    }
}

fn intMetric(comptime field: []const u8, comptime metric_name: [:0]const u8) *const fn ([]DeviceInfo) void {
    return &struct {
        fn query(infos: []DeviceInfo) void {
            var ids: [64]c_longlong = undefined;
            var vals: [64]c_longlong = undefined;
            const n = tpuinfo.queryInt(address, metric_name, &ids, &vals) catch {
                clearField(infos, field);
                return;
            };
            for (ids[0..n], vals[0..n]) |id, val| {
                if (id >= 0 and id < infos.len) {
                    @field(infos[@intCast(id)], field) = @intCast(val);
                }
            }
        }
    }.query;
}

fn doubleMetric(comptime field: []const u8, comptime metric_name: [:0]const u8) *const fn ([]DeviceInfo) void {
    return &struct {
        fn query(infos: []DeviceInfo) void {
            var ids: [64]c_longlong = undefined;
            var vals: [64]f64 = undefined;
            const n = tpuinfo.queryDouble(address, metric_name, &ids, &vals) catch {
                clearField(infos, field);
                return;
            };
            for (ids[0..n], vals[0..n]) |id, val| {
                if (id >= 0 and id < infos.len) {
                    @field(infos[@intCast(id)], field) = @intFromFloat(@round(val));
                }
            }
        }
    }.query;
}

const batch_metrics = .{
    .{ .query = intMetric("mem_used_bytes", "tpu.runtime.hbm.memory.usage.bytes") },
    .{ .query = intMetric("mem_total_bytes", "tpu.runtime.hbm.memory.total.bytes") },
    .{ .query = doubleMetric("gpu_util_percent", "tpu.runtime.tensorcore.dutycycle.percent") },
};
