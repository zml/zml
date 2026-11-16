//! The host, from /proc and /sys.
const std = @import("std");
const smi = @import("../smi.zig");

pub const Host = struct {
    io: std.Io,
    proc: ?smi.sysfs.Dir,
    /// CPU ticks of every CPU together.
    total: smi.Counter = .{},
    /// CPU ticks of each process.
    ticks: smi.Counters(u32),

    /// Reads what does not change: hostname, kernel, CPU, core count and total memory.
    pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, data: *smi.HostData) Host {
        const proc = smi.sysfs.Dir.open(io, "/proc") catch null;
        var buffer: [256]u8 = undefined;
        if (proc) |dir| {
            data.hostname = dupe(arena, dir.read("sys/kernel/hostname", &buffer) catch null);
            data.kernel = dupe(arena, dir.read("sys/kernel/osrelease", &buffer) catch null);
            data.cpu_name = dupe(arena, dir.field("cpuinfo", "model name", &buffer) catch null);
            data.mem_total_kib = smi.sysfs.firstNumber(dir.field("meminfo", "MemTotal", &buffer) catch "");
        }
        if (smi.sysfs.Dir.open(io, "/sys/devices/system/cpu")) |cpu| {
            defer cpu.close();
            data.cpu_cores = cores(cpu.read("present", &buffer) catch "");
        } else |_| {}
        return .{ .io = io, .proc = proc, .ticks = .init(gpa) };
    }

    pub fn deinit(self: *Host) void {
        self.ticks.deinit();
        if (self.proc) |proc| proc.close();
    }

    /// Overwrites what changes: available memory, load and uptime. A read that fails gives null.
    pub fn sample(self: *Host, data: *smi.HostData) void {
        data.mem_available_kib = null;
        data.load_1, data.load_5, data.load_15 = .{ null, null, null };
        data.uptime_seconds = null;
        const proc = self.proc orelse return;

        var buffer: [256]u8 = undefined;
        data.mem_available_kib = smi.sysfs.firstNumber(proc.field("meminfo", "MemAvailable", &buffer) catch "");
        if (proc.read("loadavg", &buffer)) |loadavg| {
            var words = std.mem.tokenizeScalar(u8, loadavg, ' ');
            inline for (.{ &data.load_1, &data.load_5, &data.load_15 }) |load| {
                load.* = std.fmt.parseFloat(f32, words.next() orelse "") catch null;
            }
        } else |_| {}
        if (proc.read("uptime", &buffer)) |uptime| {
            data.uptime_seconds = std.fmt.parseInt(u64, std.mem.sliceTo(uptime, '.'), 10) catch null;
        } else |_| {}
    }

    /// Adds user, command, CPU and RSS to one row per local PID.
    pub fn enrich(self: *Host, strings: std.mem.Allocator, rows: []smi.ProcessInfo) void {
        const proc = self.proc orelse return;
        defer self.ticks.sweep();

        var buffer: [512]u8 = undefined;
        const total = if (proc.head("stat", &buffer)) |stat| self.total.advance(self.io, totalTicks(stat)) else |_| null;

        for (rows) |*row| self.read(proc, strings, row, total);
    }

    fn read(self: *Host, proc: smi.sysfs.Dir, strings: std.mem.Allocator, row: *smi.ProcessInfo, total: ?smi.Growth) void {
        var path: [32]u8 = undefined;
        var buffer: [4096]u8 = undefined;

        if (proc.read(processPath(&path, row.pid, "status"), &buffer)) |status| {
            var lines = std.mem.tokenizeScalar(u8, status, '\n');
            while (lines.next()) |line| {
                if (smi.sysfs.fieldValue(line, "Uid")) |uid| row.uid = @intCast(smi.sysfs.firstNumber(uid) orelse 0);
                if (smi.sysfs.fieldValue(line, "VmRSS")) |rss| row.rss_kib = smi.sysfs.firstNumber(rss) orelse 0;
            }
            if (std.c.getpwuid(row.uid)) |user| {
                if (user.name) |name| row.username = strings.dupe(u8, std.mem.span(name)) catch "";
            }
        } else |_| {}

        if (proc.head(processPath(&path, row.pid, "cmdline"), &buffer)) |cmdline| {
            const command = buffer[0..cmdline.len];
            std.mem.replaceScalar(u8, command, 0, ' ');
            row.comm = strings.dupe(u8, std.mem.trimEnd(u8, command, " ")) catch "";
        } else |_| {}

        if (proc.read(processPath(&path, row.pid, "stat"), &buffer)) |stat| {
            const ticks = processTicks(stat) orelse return;
            const growth = (self.ticks.advance(row.pid, self.io, ticks) catch return) orelse return;
            const whole = total orelse return;
            if (whole.amount > 0) row.cpu_percent = @intCast(@min(growth.amount *| 1000 / whole.amount, std.math.maxInt(u16)));
        } else |_| {}
    }
};

fn processPath(buffer: []u8, pid: u32, file: []const u8) []const u8 {
    return std.fmt.bufPrint(buffer, "{d}/{s}", .{ pid, file }) catch "";
}

fn dupe(arena: std.mem.Allocator, string: ?[]const u8) ?[]const u8 {
    return arena.dupe(u8, string orelse return null) catch null;
}

/// The CPUs present, "0-47": the highest index, plus one.
fn cores(present: []const u8) ?u64 {
    const dash = std.mem.indexOfScalar(u8, present, '-') orelse return if (present.len > 0) 1 else null;
    return (std.fmt.parseInt(u64, present[dash + 1 ..], 10) catch return null) + 1;
}

/// The sum of the first line of /proc/stat: "cpu  user nice system idle ...".
fn totalTicks(stat: []const u8) u64 {
    var words = std.mem.tokenizeAny(u8, std.mem.sliceTo(stat, '\n'), " \t");
    _ = words.next();
    var total: u64 = 0;
    while (words.next()) |word| total +|= std.fmt.parseInt(u64, word, 10) catch 0;
    return total;
}

/// utime plus stime of /proc/<pid>/stat, the 12th and 13th fields after the command.
fn processTicks(stat: []const u8) ?u64 {
    const end = std.mem.lastIndexOfScalar(u8, stat, ')') orelse return null;
    var words = std.mem.tokenizeScalar(u8, stat[end + 1 ..], ' ');
    for (0..11) |_| _ = words.next();
    const utime = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
    const stime = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
    return utime + stime;
}
