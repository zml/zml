const std = @import("std");
const sysfs = @import("../../sysfs.zig");
const tpuinfo = @import("tpuinfo.zig");
const device_info = @import("../../info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const Worker = @import("../../worker.zig").Worker;
const pi = @import("../../info/process_info.zig");
const ProcessShadowList = @import("../../shadow_list.zig").ShadowList(pi.ProcessInfo);
const tpu_process = @import("process.zig");

const address = "localhost:8431";

pub const Backend = struct {
    processes: ProcessShadowList = .init(),

    pub fn start(self: *Backend, w: *Worker, io: std.Io, allocator: std.mem.Allocator, device_infos: *std.ArrayList(*DeviceInfo), proc_allocator: std.mem.Allocator) !void {
        const chip = scanPciChips(io) orelse return error.TpuUnavailable;
        const device_count = chip.chip_count * chip.devices_per_chip;
        const tpu_start = device_infos.items.len;

        for (0..device_count) |_| {
            const info = try allocator.create(DeviceInfo);
            info.* = .{ .tpu = .{ .name = toName(chip.name) } };
            try device_infos.append(allocator, info);
        }

        const tpu_infos = device_infos.items[tpu_start..];
        inline for (batch_metrics) |query| {
            try w.spawnBatchWorker(io, tpu_infos, query);
        }

        if (tpu_infos.len > 0) {
            try tpu_process.init(w, io, proc_allocator, chip.devices_per_chip, tpu_infos, &self.processes);
        }
    }

    pub fn deinit(self: *Backend, proc_allocator: std.mem.Allocator) void {
        self.processes.deinit(proc_allocator);
    }
};

const batch_metrics = .{
    &queryMemUsed,
    &queryMemTotal,
    &queryUtil,
};

fn queryMemUsed(infos: []*DeviceInfo) void {
    queryInt(infos, "mem_used_bytes", "tpu.runtime.hbm.memory.usage.bytes");
}

fn queryMemTotal(infos: []*DeviceInfo) void {
    queryInt(infos, "mem_total_bytes", "tpu.runtime.hbm.memory.total.bytes");
}

fn queryInt(infos: []*DeviceInfo, comptime field: []const u8, comptime metric_name: [:0]const u8) void {
    var ids: [tpuinfo.max_devices]c_longlong = undefined;
    var vals: [tpuinfo.max_devices]c_longlong = undefined;
    const n = tpuinfo.queryInt(address, metric_name, &ids, &vals) catch {
        for (infos) |info| @field(&info.tpu, field) = null;
        return;
    };
    for (ids[0..n], vals[0..n]) |id, val| {
        if (id >= 0 and id < infos.len) {
            @field(&infos[@intCast(id)].tpu, field) = @intCast(val);
        }
    }
}

fn queryUtil(infos: []*DeviceInfo) void {
    var ids: [tpuinfo.max_devices]c_longlong = undefined;
    var vals: [tpuinfo.max_devices]f64 = undefined;
    const n = tpuinfo.queryDouble(address, "tpu.runtime.tensorcore.dutycycle.percent", &ids, &vals) catch {
        for (infos) |info| info.tpu.util_percent = null;
        return;
    };
    for (ids[0..n], vals[0..n]) |id, val| {
        if (id >= 0 and id < infos.len) {
            infos[@intCast(id)].tpu.util_percent = @intFromFloat(@round(val));
        }
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

fn readSysfs(io: std.Io, path_buf: *[std.Io.Dir.max_path_bytes]u8, slot: []const u8, file: []const u8) ![256]u8 {
    const path = std.fmt.bufPrint(path_buf, pci_base ++ "/{s}/{s}", .{ slot, file }) catch return error.Overflow;

    return sysfs.readString(io, path);
}

fn sysfsEql(raw: [256]u8, expected: []const u8) bool {
    return std.mem.eql(u8, std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&raw)), 0), expected);
}

fn toName(name: []const u8) [256]u8 {
    var buf: [256]u8 = .{0} ** 256;
    @memcpy(buf[0..name.len], name);

    return buf;
}
