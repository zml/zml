const std = @import("std");
const c = @import("c");
const smi = @import("zml-smi");

const Tpu = @This();

platform: smi.Platform,
io: std.Io,
devices_per_chip: u32,

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes };

pub fn detect(io: std.Io) bool {
    return smi.sysfs.exists(io, "/dev/accel0") or smi.sysfs.exists(io, "/dev/vfio/0");
}

pub fn init(arena: std.mem.Allocator, _: std.mem.Allocator, io: std.Io) !Tpu {
    const pci = try smi.sysfs.Dir.open(io, "/sys/bus/pci/devices");
    defer pci.close();

    var found: ?Chip = null;
    var count: u32 = 0;
    var it = pci.iterate();

    while (try it.next(io)) |entry| {
        const dir = pci.openDir(entry.name) catch continue;
        defer dir.close();
        if ((dir.int(u16, "vendor") catch 0) != google) continue;
        const device = dir.int(u16, "device") catch continue;
        found = chipOf(device, dir.int(u16, "subsystem_device") catch null) orelse continue;
        count += 1;
    }
    const chip = found orelse return error.TpuUnavailable;

    const devices = try arena.alloc(smi.Device, count * chip.devices_per_chip);
    @memset(devices, .{ .target = .tpu, .name = chip.name });

    return .{
        .platform = .{ .devices = devices, .vtable = &vtable },
        .io = io,
        .devices_per_chip = chip.devices_per_chip,
    };
}

fn poll(_: *smi.Platform, _: std.mem.Allocator, samples: []smi.Sample) void {
    const used = query(c_longlong, "tpu.runtime.hbm.memory.usage.bytes");
    const total = query(c_longlong, "tpu.runtime.hbm.memory.total.bytes");
    const duty = query(f64, "tpu.runtime.tensorcore.dutycycle.percent");

    for (samples[0..@min(samples.len, max_devices)], 0..) |*sample, id| {
        sample.tpu = .{ .util_percent = duty[id], .mem_used_bytes = used[id], .mem_total_bytes = total[id] };
    }
}

fn processes(platform: *smi.Platform, _: std.mem.Allocator, samples: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
    const self: *Tpu = @alignCast(@fieldParentPtr("platform", platform));

    var holders: smi.procfs.FdHolders = .init(self.io, &.{ "/dev/accel", "/dev/vfio/" });
    defer holders.deinit();

    while (holders.next()) |holder| {
        try addHolder(out, samples, self.devices_per_chip, holder.pid, holder.rest);
    }
}

const address = "localhost:8431";
const max_devices: usize = c.TPUINFO_MAX_DEVICES;
const google = 0x1ae0;

const Chip = struct { device: u16, subsystem: ?u16 = null, name: []const u8, devices_per_chip: u32 };

/// From tpu_info/device.py in cloud-accelerator-diagnostics (c94f463).
const chips = [_]Chip{
    .{ .device = 0x0027, .subsystem = 0x004e, .name = "TPU v2", .devices_per_chip = 2 },
    .{ .device = 0x0027, .subsystem = 0x004f, .name = "TPU v3", .devices_per_chip = 2 },
    .{ .device = 0x005e, .name = "TPU v4", .devices_per_chip = 1 },
    .{ .device = 0x0063, .name = "TPU v5e", .devices_per_chip = 1 },
    .{ .device = 0x0062, .name = "TPU v5p", .devices_per_chip = 1 },
    .{ .device = 0x006f, .name = "TPU v6e", .devices_per_chip = 1 },
    .{ .device = 0x0076, .name = "TPU7x", .devices_per_chip = 2 },
};

fn chipOf(device: u16, subsystem: ?u16) ?Chip {
    for (chips) |chip| {
        if (chip.device == device and (chip.subsystem == null or chip.subsystem == subsystem)) return chip;
    }
    return null;
}

fn query(comptime T: type, metric: [:0]const u8) [max_devices]?u64 {
    var ids: [max_devices]c_longlong = undefined;
    var values: [max_devices]T = undefined;

    const count = switch (T) {
        c_longlong => c.tpu_query_int(address, metric, &ids, &values, max_devices),
        f64 => c.tpu_query_double(address, metric, &ids, &values, max_devices),
        else => @compileError("a metric is an int or a double"),
    };
    const n = std.math.cast(usize, count) orelse 0;

    return byId(T, ids[0..n], values[0..n]);
}

fn byId(comptime T: type, ids: []const c_longlong, values: []const T) [max_devices]?u64 {
    var out: [max_devices]?u64 = @splat(null);

    for (ids, values) |id, value| {
        const index = std.math.cast(usize, id) orelse continue;
        if (index >= max_devices) continue;

        out[index] = switch (T) {
            c_longlong => std.math.cast(u64, value),
            f64 => if (std.math.isFinite(value)) std.math.lossyCast(u64, @round(value)) else null,
            else => @compileError("a metric is an int or a double"),
        };
    }

    return out;
}

fn addHolder(out: *smi.ProcessList, samples: []const smi.Sample, devices_per_chip: u32, pid: u32, node: []const u8) !void {
    const chip = std.fmt.parseInt(u16, node, 10) catch return;

    for (0..devices_per_chip) |core| {
        const device = @as(usize, chip) * devices_per_chip + core;
        if (device >= samples.len) return;

        for (out.rows.items) |row| {
            if (row.pid == pid and row.device_idx == device) break;
        } else {
            const tpu = samples[device].tpu;
            try out.append(.{
                .pid = pid,
                .device_idx = @intCast(device),
                .dev_util_percent = if (tpu.util_percent) |percent| std.math.cast(u16, percent) else null,
                .dev_mem_kib = if (tpu.mem_used_bytes) |bytes| bytes / 1024 else null,
            });
        }
    }
}

test chipOf {
    try std.testing.expectEqualStrings("TPU v3", chipOf(0x0027, 0x004f).?.name);
    try std.testing.expectEqualStrings("TPU v5e", chipOf(0x0063, 0x0001).?.name);
    try std.testing.expect(chipOf(0x0027, null) == null);
    try std.testing.expect(chipOf(0x0001, null) == null);
}

test byId {
    const used = byId(c_longlong, &.{ 1, 0, -1, 64 }, &.{ 2048, -5, 7, 7 });
    try std.testing.expectEqual(2048, used[1]);
    try std.testing.expectEqual(null, used[0]);
    const duty = byId(f64, &.{ 0, 1 }, &.{ 12.5, std.math.nan(f64) });
    try std.testing.expectEqual(13, duty[0]);
    try std.testing.expectEqual(null, duty[1]);
}
