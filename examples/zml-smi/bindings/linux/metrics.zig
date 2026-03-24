const std = @import("std");
const sysfs = @import("../../sysfs.zig");
const HostInfo = @import("../../host_info.zig").HostInfo;
const worker = @import("../../worker.zig");

pub fn init(io: std.Io, info: *HostInfo, signal: *worker.Signal) !void {
    const host = Host{ .io = io };

    // Static values — read once
    info.hostname = host.getHostname() catch null;
    info.kernel = host.getKernel() catch null;
    info.cpu_name = host.getCpuName() catch null;
    info.cpu_cores = host.getCpuCores() catch null;
    info.mem_total_kib = host.getMemTotal() catch null;

    // Dynamic values — poll with workers
    inline for (dynamic_metrics) |metric| {
        try worker.spawnWorker(io, info, metric.field, metric.query, host, signal);
    }
}

const Host = struct {
    io: std.Io,

    pub fn getHostname(self: Host) ![256]u8 {
        return sysfs.readString(self.io, "/proc/sys/kernel/hostname");
    }

    pub fn getKernel(self: Host) ![256]u8 {
        return sysfs.readString(self.io, "/proc/sys/kernel/osrelease");
    }

    pub fn getCpuName(self: Host) ![256]u8 {
        return sysfs.readFieldString(self.io, "/proc/cpuinfo", "model name");
    }

    pub fn getCpuCores(self: Host) !u64 {
        var buf: [64]u8 = undefined;
        const data = try std.Io.Dir.readFile(.cwd(), self.io, "/sys/devices/system/cpu/present", &buf);
        const trimmed = std.mem.trimEnd(u8, data, &std.ascii.whitespace);
        const dash = std.mem.indexOfScalar(u8, trimmed, '-') orelse return 1;
        return (try std.fmt.parseInt(u64, trimmed[dash + 1 ..], 10)) + 1;
    }

    pub fn getMemTotal(self: Host) !u64 {
        return sysfs.readFieldInt(self.io, "/proc/meminfo", "MemTotal");
    }

    pub fn getMemAvailable(self: Host) !u64 {
        return sysfs.readFieldInt(self.io, "/proc/meminfo", "MemAvailable");
    }

    pub fn getCpuTemp(self: Host) !u64 {
        return try sysfs.readInt(self.io, "/sys/class/thermal/thermal_zone0/temp") / 1000;
    }

    pub fn getLoadAvg(self: Host) ![256]u8 {
        return sysfs.readString(self.io, "/proc/loadavg");
    }

    pub fn getUptime(self: Host) !u64 {
        var buf: [64]u8 = undefined;
        const data = try std.Io.Dir.readFile(.cwd(), self.io, "/proc/uptime", &buf);
        const dot = std.mem.indexOfScalar(u8, data, '.') orelse data.len;
        return std.fmt.parseInt(u64, data[0..dot], 10);
    }
};

const dynamic_metrics = .{
    .{ .field = "mem_available_kib", .query = Host.getMemAvailable },
    .{ .field = "cpu_temp", .query = Host.getCpuTemp },
    .{ .field = "load_avg", .query = Host.getLoadAvg },
    .{ .field = "uptime_seconds", .query = Host.getUptime },
};
