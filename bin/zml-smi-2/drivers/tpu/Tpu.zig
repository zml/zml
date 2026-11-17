//! Google TPUs, through a PCI scan and the runtime metrics service of libtpu.
const std = @import("std");
const c = @import("c");
const contract = @import("zml-smi-2/contract");
const source = @import("zml-smi-2/source");
const native = @import("zml-smi-2/native");

const Tpu = @This();

devices: []const contract.Device,
io: std.Io,
/// The service numbers the devices chip after chip.
devices_per_chip: u32,

/// Each query waits at most 1 s, so a stalled service delays a poll by up to 3 s.
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

pub const detection_paths = [_][]const u8{ "/dev/accel0", "/dev/vfio/0" };

pub fn open(arena: std.mem.Allocator, _: std.mem.Allocator, io: std.Io) !Tpu {
    var scan: native.pci.Devices = try .open(io, google);
    defer scan.close();
    var found: ?Chip = null;
    var count: u32 = 0;
    while (scan.next()) |entry| {
        defer entry.dir.close();
        found = chipOf(entry.id, entry.dir.int(u16, "subsystem_device") catch null) orelse continue;
        count += 1;
    }
    const chip = found orelse return error.TpuUnavailable;

    const devices = try arena.alloc(contract.Device, count * chip.devices_per_chip);
    @memset(devices, .{ .target = .tpu, .name = chip.name });
    return .{
        .devices = devices,
        .io = io,
        .devices_per_chip = chip.devices_per_chip,
    };
}

fn chipOf(device: u16, subsystem: ?u16) ?Chip {
    for (chips) |chip| {
        if (chip.device == device and (chip.subsystem == null or chip.subsystem == subsystem)) return chip;
    }
    return null;
}

pub fn poll(self: *Tpu, _: std.mem.Allocator, reading: *source.Reading) !void {
    const samples = reading.samples[0..@min(reading.samples.len, max_devices)];
    inline for (.{
        .{ .mem_used_bytes, "hbm.memory.usage.bytes", c_longlong },
        .{ .mem_total_bytes, "hbm.memory.total.bytes", c_longlong },
        .{ .util_percent, "tensorcore.dutycycle.percent", f64 },
    }) |spec| {
        const key, const metric, const Value = spec;
        const values = query(Value, "tpu.runtime." ++ metric);
        for (samples, values[0..samples.len]) |*sample, value| sample.put(key, value);
    }

    var holders: native.proc.FdHolders = .init(self.io, &.{ "/dev/accel", "/dev/vfio/" });
    defer holders.deinit();
    while (holders.next()) |holder| {
        try addHolder(reading, self.devices_per_chip, holder.pid, holder.rest);
    }
}

/// One metric of every device, by device id. A failed query leaves them all null.
fn query(comptime Value: type, metric: [:0]const u8) [max_devices]?u64 {
    var ids: [max_devices]c_longlong = undefined;
    var values: [max_devices]Value = undefined;
    const integers, const reals = if (Value == c_longlong) .{ &values, null } else .{ null, &values };
    const count = c.tpu_query(address, metric, &ids, integers, reals);
    const len = std.math.cast(usize, count) orelse 0;
    return byId(Value, ids[0..len], values[0..len]);
}

fn byId(comptime Value: type, ids: []const c_longlong, values: []const Value) [max_devices]?u64 {
    var out: [max_devices]?u64 = @splat(null);
    for (ids, values) |id, value| {
        const index = std.math.cast(usize, id) orelse continue;
        if (index >= max_devices) continue;
        out[index] = switch (Value) {
            c_longlong => std.math.cast(u64, value),
            f64 => if (std.math.isFinite(value)) std.math.lossyCast(u64, @round(value)) else null,
            else => @compileError("a metric is an int or a double"),
        };
    }
    return out;
}

/// Lists a process that holds the node of a chip on every device of that chip, once.
fn addHolder(reading: *source.Reading, devices_per_chip: u32, pid: u32, node: []const u8) !void {
    const chip = std.fmt.parseInt(u16, node, 10) catch return;
    for (0..devices_per_chip) |core| {
        const device = @as(usize, chip) * devices_per_chip + core;
        if (device >= reading.samples.len) return;
        for (reading.processes.items) |row| {
            if (row.pid == pid and row.device_idx == device) break;
        } else {
            const sample = &reading.samples[device];
            try reading.addProcess(.{
                .pid = pid,
                .device_idx = @intCast(device),
                .dev_util_percent = if (sample.get(.util_percent)) |percent|
                    std.math.cast(u16, percent)
                else
                    null,
                .dev_mem_kib = if (sample.get(.mem_used_bytes)) |bytes| bytes / 1024 else null,
            });
        }
    }
}

pub const testing = if (@import("builtin").is_test) struct {
    pub const chipOf = Tpu.chipOf;
    pub const byId = Tpu.byId;
    pub const addHolder = Tpu.addHolder;
} else struct {};
