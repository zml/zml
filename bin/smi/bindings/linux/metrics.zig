const std = @import("std");
const sysfs = @import("../../utils/sysfs.zig");
const host_info = @import("../../info/host_info.zig");
const HostInfo = host_info.HostInfo;
const HostData = host_info.HostData;
const Worker = @import("../../worker.zig").Worker;

pub fn init(w: *Worker, io: std.Io, info: *HostInfo) !void {
    const host: Host = .{ .io = io };

    // read once metrics
    var initial = info.get(io);
    initial.hostname = host.hostname() catch null;
    initial.kernel = host.kernel() catch null;
    initial.cpu_name = host.cpuName() catch null;
    initial.cpu_cores = host.cpuCores() catch null;
    initial.mem_total_kib = host.memTotal() catch null;
    info.set(io, initial);

    try w.spawn(io, pollHost, .{ io, w, info, host });
}

const pollHost = Worker.pollMetrics(*HostInfo, Host, metrics);

const Host = struct {
    io: std.Io,

    pub fn hostname(self: Host) ![256]u8 {
        return sysfs.readString(self.io, "/proc/sys/kernel/hostname");
    }

    pub fn kernel(self: Host) ![256]u8 {
        return sysfs.readString(self.io, "/proc/sys/kernel/osrelease");
    }

    pub fn cpuName(self: Host) ![256]u8 {
        return sysfs.readFieldString(self.io, "/proc/cpuinfo", "model name");
    }

    pub fn cpuCores(self: Host) !u64 {
        var buf: [64]u8 = undefined;

        const data = try std.Io.Dir.readFile(.cwd(), self.io, "/sys/devices/system/cpu/present", &buf);
        const trimmed = std.mem.trimEnd(u8, data, &std.ascii.whitespace);
        const dash = std.mem.indexOfScalar(u8, trimmed, '-') orelse return 1;

        return (try std.fmt.parseInt(u64, trimmed[dash + 1 ..], 10)) + 1;
    }

    pub fn memTotal(self: Host) !u64 {
        return sysfs.readFieldInt(self.io, "/proc/meminfo", "MemTotal");
    }

    pub fn memAvailable(self: Host) !u64 {
        return sysfs.readFieldInt(self.io, "/proc/meminfo", "MemAvailable");
    }

    pub fn loadAvg(self: Host) ![256]u8 {
        return sysfs.readString(self.io, "/proc/loadavg");
    }

    pub fn uptime(self: Host) !u64 {
        var buf: [64]u8 = undefined;

        const data = try std.Io.Dir.readFile(.cwd(), self.io, "/proc/uptime", &buf);
        const dot = std.mem.indexOfScalar(u8, data, '.') orelse data.len;

        return std.fmt.parseInt(u64, data[0..dot], 10);
    }
};

const metrics = .{
    .{ .field = "mem_available_kib", .query = Host.memAvailable },
    .{ .field = "load_avg", .query = Host.loadAvg },
    .{ .field = "uptime_seconds", .query = Host.uptime },
};
