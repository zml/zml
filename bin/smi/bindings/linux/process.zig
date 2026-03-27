const std = @import("std");
const sysfs = @import("../../utils/sysfs.zig");
const pi = @import("../../info/process_info.zig");

const ProcessInfo = pi.ProcessInfo;

pub const ProcessEnricher = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    prev_ticks: std.AutoHashMapUnmanaged(u32, u64) = .{},
    prev_total_ticks: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io) !ProcessEnricher {
        var enricher: ProcessEnricher = .{
            .gpa = gpa,
            .arena = std.heap.ArenaAllocator.init(gpa),
            .prev_total_ticks = readTotalCpuTicks(io),
            .prev_ticks = .{},
        };

        try enricher.prev_ticks.ensureTotalCapacity(gpa, 64);

        return enricher;
    }

    pub fn deinit(self: *ProcessEnricher) void {
        self.prev_ticks.deinit(self.gpa);
        self.arena.deinit();
    }

    pub fn enrich(self: *ProcessEnricher, io: std.Io, procs: []ProcessInfo) void {
        _ = self.arena.reset(.retain_capacity);
        const alloc = self.arena.allocator();

        const curr_total = readTotalCpuTicks(io);
        const delta_total = curr_total -| self.prev_total_ticks;

        var curr_ticks: std.AutoHashMapUnmanaged(u32, u64) = .{};
        curr_ticks.ensureTotalCapacity(self.gpa, @intCast(procs.len)) catch return;

        for (procs) |*info| {
            var path_buf: [4096]u8 = undefined;

            // Read Uid, VmRSS from /proc/{pid}/status
            const status_path = std.fmt.bufPrint(&path_buf, "/proc/{d}/status", .{info.pid}) catch continue;
            const uid: u32 = @intCast(sysfs.readFieldInt(io, status_path, "Uid") catch 0);
            info.uid = uid;
            info.username = lookupUsername(alloc, uid) catch continue;
            info.rss_kib = sysfs.readFieldInt(io, status_path, "VmRSS") catch 0;

            // Read cmdline
            const cmdline_path = std.fmt.bufPrint(&path_buf, "/proc/{d}/cmdline", .{info.pid}) catch continue;
            info.comm = readCmdline(alloc, io, cmdline_path) catch continue;

            // CPU% delta
            const ticks = readProcTicks(io, info.pid) orelse continue;
            if (delta_total > 0) {
                if (self.prev_ticks.get(info.pid)) |prev| {
                    const delta_proc = ticks -| prev;
                    info.cpu_percent = @intCast(@min(delta_proc * 1000 / delta_total, std.math.maxInt(u16)));
                }
            }

            curr_ticks.getOrPutAssumeCapacity(info.pid).value_ptr.* = ticks;
        }

        self.prev_ticks.deinit(self.gpa);
        self.prev_ticks = curr_ticks;
        self.prev_total_ticks = curr_total;
    }
};

fn lookupUsername(allocator: std.mem.Allocator, uid: u32) ![]const u8 {
    const pw = std.c.getpwuid(uid) orelse return error.NoUser;
    const name = std.mem.span(pw.name orelse return error.NoName);
    return try allocator.dupe(u8, name);
}

/// Extract utime + stime from /proc/{pid}/stat.
fn readProcTicks(io: std.Io, pid: u32) ?u64 {
    var path_buf: [4096]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/proc/{d}/stat", .{pid}) catch return null;
    var buf: [1024]u8 = undefined;
    const data = std.Io.Dir.readFile(.cwd(), io, path, &buf) catch return null;

    // Fields after ") ": state=0, ppid=1, ..., utime=11, stime=12
    const rest = data[(std.mem.lastIndexOfScalar(u8, data, ')') orelse return null) + 2 ..];

    return sysfs.nthTokenInt(rest, 11) + sysfs.nthTokenInt(rest, 12);
}

fn readCmdline(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    var buf: [4096]u8 = undefined;
    const data = try std.Io.Dir.readFile(.cwd(), io, path, &buf);

    for (data) |*c| {
        if (c.* == 0) {
            c.* = ' ';
        }
    }

    const trimmed = std.mem.trimEnd(u8, data, &std.ascii.whitespace);
    return allocator.dupe(u8, trimmed);
}

fn readTotalCpuTicks(io: std.Io) u64 {
    var buf: [512]u8 = undefined;
    const data = std.Io.Dir.readFile(.cwd(), io, "/proc/stat", &buf) catch return 0;
    const line = data[0 .. std.mem.indexOfScalar(u8, data, '\n') orelse data.len];

    var total: u64 = 0;
    var iter = std.mem.tokenizeAny(u8, line, " \t");

    _ = iter.next(); // skip "cpu"
    while (iter.next()) |tok| {
        total += std.fmt.parseInt(u64, tok, 10) catch 0;
    }

    return total;
}
