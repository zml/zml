const std = @import("std");
const Monitor = @import("monitor.zig");

const OneApi = @This();

pub const Handle = Monitor.Handle;
pub const ProcessInfo = Monitor.ProcessInfo;

monitor: Monitor,

pub fn init(allocator: std.mem.Allocator, io: std.Io) !OneApi {
    return .{
        .monitor = try Monitor.init(allocator, io),
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
    return self.monitor.driverVersion(allocator);
}

pub fn powerUsage(self: *OneApi, handle: Handle) !u64 {
    return self.monitor.powerUsage(handle);
}

pub fn powerLimit(self: *OneApi, handle: Handle) !u64 {
    return self.monitor.powerLimit(handle);
}

pub fn temperature(self: *OneApi, handle: Handle) !u64 {
    return self.monitor.temperature(handle);
}

pub fn gpuUtil(self: *OneApi, handle: Handle) !u64 {
    return self.monitor.gpuUtil(handle);
}

pub fn clockGraphics(self: *OneApi, handle: Handle) !u64 {
    return self.monitor.clockGraphics(handle);
}

pub fn maxClockGraphics(self: *OneApi, handle: Handle) !u64 {
    return self.monitor.maxClockGraphics(handle);
}

pub fn memUsed(self: *OneApi, handle: Handle) !u64 {
    return self.monitor.memUsed(handle);
}

pub fn memTotal(self: *OneApi, handle: Handle) !u64 {
    return self.monitor.memTotal(handle);
}

pub fn pcieLinkGen(self: *OneApi, handle: Handle) !u64 {
    return self.monitor.pcieLinkGen(handle);
}

pub fn pcieLinkWidth(self: *OneApi, handle: Handle) !u64 {
    return self.monitor.pcieLinkWidth(handle);
}

pub fn pcieBandwidth(self: *OneApi, handle: Handle) !u64 {
    return self.monitor.pcieBandwidth(handle);
}

pub fn processList(self: *OneApi, allocator: std.mem.Allocator) !std.ArrayList(ProcessInfo) {
    return self.monitor.processList(allocator);
}
