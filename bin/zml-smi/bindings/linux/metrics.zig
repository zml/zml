const std = @import("std");
const sysfs = @import("zml-smi/utils").sysfs;
const host_info = @import("zml-smi/info").host_info;
const HostInfo = host_info.HostInfo;
const HostData = host_info.HostData;
const Worker = @import("zml-smi/worker").Worker;
const Collector = @import("zml-smi/collector").Collector;

pub fn init(w: *Worker, io: std.Io, collector: *Collector, info: *HostInfo) !void {
    const poll_arena = try collector.createPollArena();
    const host: Host = .{ .io = io, .arena = poll_arena };

    // read once metrics
    var initial = info.front().*;
    initial.hostname = host.hostname(collector.arena) catch null;
    initial.kernel = host.kernel(collector.arena) catch null;
    initial.cpu_name = host.cpuName(collector.arena) catch null;
    initial.cpu_cores = host.cpuCores() catch null;
    initial.mem_total_kib = host.memTotal() catch null;
    info.back().* = initial;
    info.swap();

    try w.spawn(io, pollHost, .{ io, w, info, host });
}

const pollHost = Worker.pollMetrics(*HostInfo, Host, metrics);

const Host = struct {
    io: std.Io,
    arena: *std.heap.ArenaAllocator,

    pub fn hostname(self: Host, allocator: std.mem.Allocator) ![]const u8 {
        return sysfs.readString(allocator, self.io, "/proc/sys/kernel/hostname");
    }

    pub fn kernel(self: Host, allocator: std.mem.Allocator) ![]const u8 {
        return sysfs.readString(allocator, self.io, "/proc/sys/kernel/osrelease");
    }

    pub fn cpuName(self: Host, allocator: std.mem.Allocator) ![]const u8 {
        return sysfs.readFieldString(allocator, self.io, "/proc/cpuinfo", "model name");
    }

    pub fn cpuCores(self: Host) !u64 {
        const data = try sysfs.readString(self.arena.allocator(), self.io, "/sys/devices/system/cpu/present");
        const dash = std.mem.indexOfScalar(u8, data, '-') orelse return 1;

        return (try std.fmt.parseInt(u64, data[dash + 1 ..], 10)) + 1;
    }

    pub fn memTotal(self: Host) !u64 {
        return sysfs.readFieldInt(self.arena.allocator(), self.io, "/proc/meminfo", "MemTotal");
    }

    pub fn memAvailable(self: Host) !u64 {
        return sysfs.readFieldInt(self.arena.allocator(), self.io, "/proc/meminfo", "MemAvailable");
    }

    fn parseLoadAvg(self: Host, comptime n: usize) !f32 {
        const data = try sysfs.readString(self.arena.allocator(), self.io, "/proc/loadavg");
        var iter = std.mem.splitScalar(u8, data, ' ');

        var i: usize = 0;
        while (iter.next()) |tok| : (i += 1) {
            if (i == n) return std.fmt.parseFloat(f32, tok);
        }

        return error.NotFound;
    }

    pub fn load1(self: Host) !f32 {
        return self.parseLoadAvg(0);
    }

    pub fn load5(self: Host) !f32 {
        return self.parseLoadAvg(1);
    }

    pub fn load15(self: Host) !f32 {
        return self.parseLoadAvg(2);
    }

    pub fn uptime(self: Host) !u64 {
        const data = try sysfs.readString(self.arena.allocator(), self.io, "/proc/uptime");
        const dot = std.mem.indexOfScalar(u8, data, '.') orelse data.len;

        return std.fmt.parseInt(u64, data[0..dot], 10);
    }
};

const metrics = .{
    .{ .field = "mem_available_kib", .query = Host.memAvailable },
    .{ .field = "load_1", .query = Host.load1 },
    .{ .field = "load_5", .query = Host.load5 },
    .{ .field = "load_15", .query = Host.load15 },
    .{ .field = "uptime_seconds", .query = Host.uptime },
};
