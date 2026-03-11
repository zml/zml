const std = @import("std");
const sysfs = @import("../../sysfs.zig");
const pi = @import("../../info/process_info.zig");

const ProcessInfo = pi.ProcessInfo;

pub const ProcessEnricher = struct {
    gpa: std.mem.Allocator,
    prev_ticks: std.AutoHashMapUnmanaged(u32, u64) = .{},
    prev_total_ticks: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io) !ProcessEnricher {
        var enricher: ProcessEnricher = .{
            .gpa = gpa,
            .prev_total_ticks = readTotalCpuTicks(io),
            .prev_ticks = .{},
        };

        try enricher.prev_ticks.ensureTotalCapacity(gpa, 64);

        return enricher;
    }

    pub fn enrich(self: *ProcessEnricher, io: std.Io, procs: []ProcessInfo) void {
        const curr_total = readTotalCpuTicks(io);
        const delta_total = curr_total -| self.prev_total_ticks;

        var curr_ticks: std.AutoHashMapUnmanaged(u32, u64) = .{};
        curr_ticks.ensureTotalCapacity(self.gpa, @intCast(procs.len)) catch {};

        for (procs) |*info| {
            var path_buf: [64]u8 = undefined;

            // Read Uid, VmRSS, Name from /proc/{pid}/status
            const status_path = std.fmt.bufPrint(&path_buf, "/proc/{d}/status", .{info.pid}) catch continue;
            const uid: u32 = @intCast(sysfs.readFieldInt(io, status_path, "Uid") catch 0);
            info.uid = uid;
            info.username = lookupUsername(uid);
            info.rss_kib = sysfs.readFieldInt(io, status_path, "VmRSS") catch 0;
            const name = sysfs.readFieldString(io, status_path, "Name") catch .{0} ** 256;

            // Read cmdline, fall back to Name from status
            const cmdline_path = std.fmt.bufPrint(&path_buf, "/proc/{d}/cmdline", .{info.pid}) catch continue;
            const cmdline = readCmdline(io, cmdline_path);
            info.comm = if (cmdline[0] != 0) cmdline else name;

            // CPU% delta
            const ticks = readProcTicks(io, info.pid) orelse continue;
            if (delta_total > 0) {
                if (self.prev_ticks.get(info.pid)) |prev| {
                    const delta_proc = ticks -| prev;
                    info.cpu_percent = @intCast(@min(delta_proc * 1000 / delta_total, 9999));
                }
            }

            curr_ticks.getOrPutAssumeCapacity(info.pid).value_ptr.* = ticks;
        }

        self.prev_ticks.deinit(self.gpa);
        self.prev_ticks = curr_ticks;
        self.prev_total_ticks = curr_total;
    }
};

fn lookupUsername(uid: u32) [32]u8 {
    if (std.c.getpwuid(uid)) |pw| {
        const name: [*:0]const u8 = pw.name orelse return formatUid(uid);
        var buf: [32]u8 = .{0} ** 32;
        const len = @min(std.mem.len(name), 31);
        @memcpy(buf[0..len], name[0..len]);
        return buf;
    }

    return formatUid(uid);
}

fn formatUid(uid: u32) [32]u8 {
    var buf: [32]u8 = .{0} ** 32;
    _ = std.fmt.bufPrint(&buf, "{d}", .{uid}) catch {};

    return buf;
}

/// Extract utime + stime from /proc/{pid}/stat.
fn readProcTicks(io: std.Io, pid: u32) ?u64 {
    var path_buf: [64]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/proc/{d}/stat", .{pid}) catch return null;
    var buf: [1024]u8 = undefined;
    const data = std.Io.Dir.readFile(.cwd(), io, path, &buf) catch return null;

    // Fields after ") ": state=0, ppid=1, ..., utime=11, stime=12
    const rest = data[(std.mem.lastIndexOfScalar(u8, data, ')') orelse return null) + 2 ..];

    return sysfs.nthTokenInt(rest, 11) + sysfs.nthTokenInt(rest, 12);
}

fn readCmdline(io: std.Io, path: []const u8) [256]u8 {
    var buf: [256]u8 = .{0} ** 256;
    const data = std.Io.Dir.readFile(.cwd(), io, path, &buf) catch return .{0} ** 256;
    const len = std.mem.trimEnd(u8, data, &.{0}).len;

    for (buf[0..len]) |*c| {
        if (c.* == 0) c.* = ' ';
    }
    if (len < 256) @memset(buf[len..], 0);

    return buf;
}

fn readTotalCpuTicks(io: std.Io) u64 {
    var buf: [512]u8 = undefined;
    const data = std.Io.Dir.readFile(.cwd(), io, "/proc/stat", &buf) catch return 0;
    const line = data[0 .. std.mem.indexOfScalar(u8, data, '\n') orelse data.len];

    var total: u64 = 0;
    var iter = std.mem.tokenizeAny(u8, line, " \t");

    _ = iter.next(); // skip "cpu"
    while (iter.next()) |tok| total += std.fmt.parseInt(u64, tok, 10) catch 0;

    return total;
}
