const std = @import("std");
const sysfs = @import("zml-smi/sysfs");
const pi = @import("zml-smi/info").process_info;

const ProcessInfo = pi.ProcessInfo;

pub const ProcessEnricher = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    prev_ticks: std.AutoHashMapUnmanaged(u32, u64) = .{},
    prev_total_ticks: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io) !ProcessEnricher {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();

        const prev_total_ticks = readTotalCpuTicks(arena.allocator(), io);
        var enricher: ProcessEnricher = .{
            .gpa = gpa,
            .arena = arena,
            .prev_total_ticks = prev_total_ticks,
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

        const curr_total = readTotalCpuTicks(alloc, io);
        const delta_total = curr_total -| self.prev_total_ticks;

        var curr_ticks: std.AutoHashMapUnmanaged(u32, u64) = .{};
        curr_ticks.ensureTotalCapacity(self.gpa, @intCast(procs.len)) catch return;

        for (procs) |*info| {
            // Read Uid, VmRSS from /proc/{pid}/status
            const status_path = std.fmt.allocPrint(alloc, "/proc/{d}/status", .{info.pid}) catch continue;
            const uid: u32 = @intCast(sysfs.readFieldInt(alloc, io, status_path, "Uid") catch 0);
            info.uid = uid;
            info.username = lookupUsername(alloc, uid) catch continue;
            info.rss_kib = sysfs.readFieldInt(alloc, io, status_path, "VmRSS") catch 0;

            // Read cmdline
            const cmdline_path = std.fmt.allocPrint(alloc, "/proc/{d}/cmdline", .{info.pid}) catch continue;
            info.comm = readCmdline(alloc, io, cmdline_path) catch continue;

            // CPU% delta
            const ticks = readProcTicks(alloc, io, info.pid) orelse continue;
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
fn readProcTicks(allocator: std.mem.Allocator, io: std.Io, pid: u32) ?u64 {
    const path = std.fmt.allocPrint(allocator, "/proc/{d}/stat", .{pid}) catch return null;
    const data = sysfs.readFirstLine(allocator, io, path) catch return null;

    // Fields after ") ": state=0, ppid=1, ..., utime=11, stime=12
    const rest = data[(std.mem.lastIndexOfScalar(u8, data, ')') orelse return null) + 2 ..];

    return sysfs.nthTokenInt(rest, 11) + sysfs.nthTokenInt(rest, 12);
}

fn readCmdline(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    const data = try sysfs.readFirstLine(allocator, io, path);

    for (data) |*c| {
        if (c.* == 0) {
            c.* = ' ';
        }
    }

    return std.mem.trimEnd(u8, data, &std.ascii.whitespace);
}

fn readTotalCpuTicks(allocator: std.mem.Allocator, io: std.Io) u64 {
    const data = sysfs.readFirstLine(allocator, io, "/proc/stat") catch return 0;

    var total: u64 = 0;
    var iter = std.mem.tokenizeAny(u8, data, " \t");

    _ = iter.next(); // skip "cpu"
    while (iter.next()) |tok| {
        total += std.fmt.parseInt(u64, tok, 10) catch 0;
    }

    return total;
}
