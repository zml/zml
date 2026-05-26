const std = @import("std");
const sysfs = @import("sysfs.zig");

const OneApi = @This();

pub const Handle = sysfs.Handle;
pub const Target = sysfs.Target;
pub const EngineSample = sysfs.EngineSample;
pub const EnergySample = sysfs.EnergySample;
pub const DeviceSample = sysfs.DeviceSample;

io: std.Io,
handles: []Handle,

pub fn init(allocator: std.mem.Allocator, io: std.Io) !OneApi {
    return .{
        .io = io,
        .handles = try sysfs.discoverIntelDevices(allocator, io),
    };
}

pub fn deviceCount(self: OneApi) usize {
    return self.handles.len;
}

pub fn handleByIndex(self: OneApi, device_id: usize) !Handle {
    if (device_id >= self.handles.len) return error.not_found;
    return self.handles[device_id];
}

pub fn target(self: OneApi, handle: Handle, device_idx: u16) Target {
    _ = self;
    return .{ .device_idx = device_idx, .bus_device_function = handle.bus_device_function };
}

pub fn name(self: OneApi, allocator: std.mem.Allocator, handle: Handle) ![]const u8 {
    const id = try sysfs.deviceId(allocator, self.io, handle);
    if (std.ascii.eqlIgnoreCase(id, "0xe223")) {
        return try allocator.dupe(u8, "Intel(R) Arc(TM) Pro B70 Graphics");
    }
    return try std.fmt.allocPrint(allocator, "Intel GPU {s}", .{id});
}

pub fn driverVersion(self: OneApi, allocator: std.mem.Allocator) ![]const u8 {
    return sysfs.driverVersion(allocator, self.io);
}

pub fn temperature(self: OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    return sysfs.temperature(allocator, self.io, handle);
}

pub fn energy(self: OneApi, allocator: std.mem.Allocator, handle: Handle) !EnergySample {
    return sysfs.energy(allocator, self.io, handle);
}

pub fn powerLimit(self: OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    return sysfs.powerLimit(allocator, self.io, handle);
}

pub fn clockGraphics(self: OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    return sysfs.clockGraphics(allocator, self.io, handle);
}

pub fn maxClockGraphics(self: OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    return sysfs.maxClockGraphics(allocator, self.io, handle);
}

pub fn memTotal(self: OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    return sysfs.memTotal(allocator, self.io, handle);
}

pub fn pcieLinkGen(self: OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    return sysfs.pcieLinkGen(allocator, self.io, handle);
}

pub fn pcieLinkWidth(self: OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    return sysfs.pcieLinkWidth(allocator, self.io, handle);
}

pub fn pcieBandwidth(self: OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    return sysfs.pcieBandwidth(allocator, self.io, handle);
}
