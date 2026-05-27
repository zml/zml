const std = @import("std");
const Monitor = @import("monitor.zig");
const smi_sysfs = @import("zml-smi/sysfs");

const OneApi = @This();

pub const Handle = Monitor.Handle;
pub const ProcessInfo = Monitor.ProcessInfo;

monitor: Monitor,
io: std.Io,

pub fn init(allocator: std.mem.Allocator, io: std.Io) !OneApi {
    return .{
        .monitor = try Monitor.init(allocator, io),
        .io = io,
    };
}

pub fn deviceCount(self: *const OneApi) usize {
    return self.monitor.deviceCount();
}

pub fn handleByIndex(self: *const OneApi, device_id: usize) !Handle {
    return self.monitor.handleByIndex(device_id);
}

pub fn name(self: *const OneApi, allocator: std.mem.Allocator, handle: Handle) ![]const u8 {
    const id = try self.monitor.deviceId(allocator, handle);
    if (std.ascii.eqlIgnoreCase(id, "0xe223")) {
        return try allocator.dupe(u8, "Intel(R) Arc(TM) Pro B70 Graphics");
    }
    return try std.fmt.allocPrint(allocator, "Intel GPU {s}", .{id});
}

pub fn driverVersion(self: *const OneApi, allocator: std.mem.Allocator) ![]const u8 {
    const version = smi_sysfs.readString(allocator, self.io, "/sys/module/xe/version") catch
        smi_sysfs.readString(allocator, self.io, "/sys/module/xe/srcversion") catch
        smi_sysfs.readString(allocator, self.io, "/sys/module/i915/version") catch
        try smi_sysfs.readString(allocator, self.io, "/sys/module/i915/srcversion");
    const trimmed = std.mem.trim(u8, version, &std.ascii.whitespace);
    if (trimmed.len == 0) return error.not_found;
    return trimmed;
}

pub fn powerUsage(self: *OneApi, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    const allocator = block: {
        _ = dev.arena.reset(.retain_capacity);
        break :block dev.arena.allocator();
    };
    const current = Monitor.readEnergy(allocator, self.io, dev.*) catch |err| {
        dev.energy_prev = null;
        return err;
    };
    const power = Monitor.powerMilliwatts(dev.energy_prev, current) orelse {
        dev.energy_prev = current;
        return error.not_found;
    };
    dev.energy_prev = current;
    return power;
}

pub fn powerLimit(self: *OneApi, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    const allocator = block: {
        _ = dev.arena.reset(.retain_capacity);
        break :block dev.arena.allocator();
    };
    return Monitor.readPowerLimit(allocator, self.io, dev.*);
}

pub fn temperature(self: *OneApi, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    const allocator = block: {
        _ = dev.arena.reset(.retain_capacity);
        break :block dev.arena.allocator();
    };
    return Monitor.readTemperature(allocator, self.io, dev.*);
}

pub fn gpuUtil(self: *OneApi, handle: Handle) !u64 {
    const dev_idx: u16 = @intCast(@intFromEnum(handle));
    const dev = try self.monitor.device(handle);
    const allocator = block: {
        _ = dev.arena.reset(.retain_capacity);
        break :block dev.arena.allocator();
    };

    var usage = Monitor.collectDeviceUsage(allocator, self.io, self.monitor.devices) catch {
        dev.activity_prev = null;
        return 0;
    };
    defer usage.deinit(allocator);

    const current = if (usage.get(dev_idx)) |sample| sample.engine else null;
    const util = Monitor.processUtil(dev.activity_prev, current) orelse 0;
    dev.activity_prev = current;
    return util;
}

pub fn clockGraphics(self: *OneApi, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    const allocator = block: {
        _ = dev.arena.reset(.retain_capacity);
        break :block dev.arena.allocator();
    };
    return Monitor.readClockGraphics(allocator, self.io, dev.*);
}

pub fn maxClockGraphics(self: *OneApi, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    const allocator = block: {
        _ = dev.arena.reset(.retain_capacity);
        break :block dev.arena.allocator();
    };
    return Monitor.readMaxClockGraphics(allocator, self.io, dev.*);
}

pub fn memUsed(self: *OneApi, handle: Handle) !u64 {
    const dev_idx: u16 = @intCast(@intFromEnum(handle));
    const dev = try self.monitor.device(handle);
    const allocator = block: {
        _ = dev.arena.reset(.retain_capacity);
        break :block dev.arena.allocator();
    };

    var usage = Monitor.collectDeviceUsage(allocator, self.io, self.monitor.devices) catch return 0;
    defer usage.deinit(allocator);

    return if (usage.get(dev_idx)) |sample| sample.mem_kib * 1024 else 0;
}

pub fn memTotal(self: *OneApi, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    const allocator = block: {
        _ = dev.arena.reset(.retain_capacity);
        break :block dev.arena.allocator();
    };
    return Monitor.readMemTotal(allocator, self.io, dev.*);
}

pub fn pcieLinkGen(self: *OneApi, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    const allocator = block: {
        _ = dev.arena.reset(.retain_capacity);
        break :block dev.arena.allocator();
    };
    return Monitor.readPcieLinkGen(allocator, self.io, dev.*);
}

pub fn pcieLinkWidth(self: *OneApi, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    const allocator = block: {
        _ = dev.arena.reset(.retain_capacity);
        break :block dev.arena.allocator();
    };
    return Monitor.readPcieLinkWidth(allocator, self.io, dev.*);
}

pub fn pcieBandwidth(self: *OneApi, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    const allocator = block: {
        _ = dev.arena.reset(.retain_capacity);
        break :block dev.arena.allocator();
    };
    return Monitor.readPcieBandwidth(allocator, self.io, dev.*);
}

pub fn processList(self: *OneApi, allocator: std.mem.Allocator) !std.ArrayList(ProcessInfo) {
    var usage = try Monitor.collectProcessUsage(allocator, self.io, self.monitor.devices);
    defer usage.deinit(allocator);

    var processes: std.ArrayList(ProcessInfo) = .empty;
    errdefer processes.deinit(allocator);

    var it = usage.iterator();
    while (it.next()) |entry| {
        const sample = entry.value_ptr.*;
        try processes.append(allocator, .{
            .pid = sample.pid,
            .device_idx = sample.device_idx,
            .mem_kib = sample.mem_kib,
            .util_percent = Monitor.processUtil(self.monitor.process_previous.get(entry.key_ptr.*), sample.engine),
        });
    }

    self.monitor.saveProcessPrevious(&usage) catch {};
    return processes;
}
