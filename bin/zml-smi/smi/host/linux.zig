const std = @import("std");
const smi = @import("../smi.zig");

pub const Host = struct {
    io: std.Io,
    gpa: std.mem.Allocator,
    proc: ?smi.sysfs.Dir,
    last_total: ?u64 = null,
    prev_ticks: std.AutoHashMapUnmanaged(u32, u64) = .empty,
    curr_ticks: std.AutoHashMapUnmanaged(u32, u64) = .empty,

    pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, data: *smi.HostData) Host {
        const uts = std.posix.uname();
        data.hostname = arena.dupe(u8, std.mem.sliceTo(&uts.nodename, 0)) catch null;
        data.kernel = arena.dupe(u8, std.mem.sliceTo(&uts.release, 0)) catch null;
        data.cpu_cores = std.Thread.getCpuCount() catch null;

        const proc = smi.sysfs.Dir.open(io, "/proc") catch null;
        var buffer: [256]u8 = undefined;
        if (proc) |dir| {
            if (dir.field("cpuinfo", "model name", &buffer)) |name| {
                data.cpu_name = arena.dupe(u8, name) catch null;
            } else |_| {}
            data.mem_total_kib = smi.sysfs.firstNumber(dir.field("meminfo", "MemTotal", &buffer) catch "");
        }

        return .{ .io = io, .gpa = gpa, .proc = proc };
    }

    pub fn deinit(self: *Host) void {
        self.prev_ticks.deinit(self.gpa);
        self.curr_ticks.deinit(self.gpa);
        if (self.proc) |proc| proc.close();
    }

    pub fn sample(self: *Host, data: *smi.HostData) void {
        data.mem_available_kib = null;
        data.load_1, data.load_5, data.load_15 = .{ null, null, null };
        data.uptime_seconds = null;

        var info: std.os.linux.Sysinfo = undefined;
        if (std.os.linux.errno(std.os.linux.sysinfo(&info)) == .SUCCESS) {
            const si_load_shift = 16;
            inline for (.{ &data.load_1, &data.load_5, &data.load_15 }, 0..) |load, i| {
                load.* = @as(f32, @floatFromInt(info.loads[i])) / @as(f32, 1 << si_load_shift);
            }

            data.uptime_seconds = std.math.cast(u64, info.uptime);
        }

        const proc = self.proc orelse return;
        var buffer: [256]u8 = undefined;
        data.mem_available_kib = smi.sysfs.firstNumber(proc.field("meminfo", "MemAvailable", &buffer) catch "");
    }

    pub fn enrich(self: *Host, strings: std.mem.Allocator, scratch: std.mem.Allocator, rows: []smi.ProcessInfo) void {
        const proc = self.proc orelse return;
        self.curr_ticks.clearRetainingCapacity();
        defer std.mem.swap(@TypeOf(self.prev_ticks), &self.prev_ticks, &self.curr_ticks);

        var buffer: [512]u8 = undefined;
        const total: ?u64 = blk: {
            const stat = proc.handle.readFile(self.io, "stat", &buffer) catch break :blk null;
            const current = totalTicks(stat);
            const prev = self.last_total;
            self.last_total = current;

            const p = prev orelse break :blk null;
            if (current < p) break :blk null;

            break :blk current - p;
        };

        var first_row: std.AutoHashMapUnmanaged(u32, usize) = .empty;
        for (rows, 0..) |*row, i| {
            if (row.remote) continue;
            const entry = first_row.getOrPut(scratch, row.pid) catch continue;

            if (entry.found_existing) {
                const done = rows[entry.value_ptr.*];
                row.uid, row.username, row.comm, row.cpu_percent, row.rss_kib = .{ done.uid, done.username, done.comm, done.cpu_percent, done.rss_kib };
                continue;
            }

            entry.value_ptr.* = i;
            self.read(proc, strings, row, total);
        }
    }

    fn read(self: *Host, proc: smi.sysfs.Dir, strings: std.mem.Allocator, row: *smi.ProcessInfo, total: ?u64) void {
        var path: [16]u8 = undefined;
        const pid_dir = proc.openDir(std.fmt.bufPrint(&path, "{d}", .{row.pid}) catch return) catch return;
        defer pid_dir.close();

        var buffer: [4096]u8 = undefined;
        if (pid_dir.field("status", "Uid", &buffer)) |uid| {
            row.uid = std.math.cast(u32, smi.sysfs.firstNumber(uid) orelse 0) orelse 0;
            var pwd: std.c.passwd = undefined;
            var result: ?*std.c.passwd = null;
            if (std.c.getpwuid_r(row.uid, &pwd, &buffer, buffer.len, &result) == 0) {
                if (result) |user| {
                    if (user.name) |name| row.username = strings.dupe(u8, std.mem.span(name)) catch "";
                }
            }
        } else |_| {}

        if (pid_dir.handle.readFile(self.io, "cmdline", &buffer)) |command| {
            std.mem.replaceScalar(u8, command, 0, ' ');
            row.comm = strings.dupe(u8, std.mem.trimEnd(u8, command, " ")) catch "";
        } else |_| {}

        if (pid_dir.read("stat", &buffer)) |stat| {
            const parsed = parseStat(stat) orelse return;
            row.rss_kib = parsed.rss_pages * (std.heap.pageSize() / 1024);
            self.curr_ticks.put(self.gpa, row.pid, parsed.ticks) catch return;
            const prev = self.prev_ticks.get(row.pid) orelse return;
            const whole = total orelse return;
            if (parsed.ticks >= prev and whole > 0) {
                row.cpu_percent = @intCast(@min((parsed.ticks - prev) *| 1000 / whole, std.math.maxInt(u16)));
            }
        } else |_| {}
    }
};

fn totalTicks(stat: []const u8) u64 {
    var words = std.mem.tokenizeAny(u8, std.mem.sliceTo(stat, '\n'), " \t");
    _ = words.next();

    var total: u64 = 0;
    while (words.next()) |word| {
        total +|= std.fmt.parseInt(u64, word, 10) catch 0;
    }

    return total;
}

/// utime + stime (fields 14-15) and rss in pages (field 24) of /proc/<pid>/stat.
fn parseStat(stat: []const u8) ?struct { ticks: u64, rss_pages: u64 } {
    _, const after = std.mem.cutScalarLast(u8, stat, ')') orelse return null;
    var words = std.mem.tokenizeAny(u8, after, " \t\r\n");

    for (0..11) |_| _ = words.next();
    const utime = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
    const stime = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;

    for (0..8) |_| _ = words.next();
    const rss_pages = std.fmt.parseInt(u64, words.next() orelse "", 10) catch 0;

    return .{ .ticks = utime +| stime, .rss_pages = rss_pages };
}

test parseStat {
    const parsed = parseStat("1234 (worker (gpu) 0) S 1 1234 1234 0 -1 4194304 100 0 10 0 250 150 0 0 20 0 4 0 9999 1048576 512 18446744073709551615").?;
    try std.testing.expectEqual(400, parsed.ticks);
    try std.testing.expectEqual(512, parsed.rss_pages);
    try std.testing.expectEqual(null, parseStat("1234 bad"));
}
