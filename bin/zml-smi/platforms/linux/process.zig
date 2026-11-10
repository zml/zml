const std = @import("std");
const sysfs = @import("zml-smi/sysfs");
const pi = @import("zml-smi/info").process_info;
const Collector = @import("zml-smi/collector").Collector;

const ProcessInfo = pi.ProcessInfo;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(ProcessInfo));

/// Merges every process list into `collector.processes`, completed with what
/// the host knows about each local process.
pub fn start(collector: *Collector) !void {
    const enricher = try collector.arena.create(ProcessEnricher);
    enricher.* = .{ .arenas = .{ try collector.createPollArena(), try collector.createPollArena() } };
    try collector.spawnPoll(pollOnce, .{ enricher, collector.gpa, collector.io, collector.process_lists.items, &collector.processes });
}

fn pollOnce(enricher: *ProcessEnricher, gpa: std.mem.Allocator, io: std.Io, lists: []const *const ProcessDoubleBuffer, processes: *ProcessDoubleBuffer) void {
    // Each side of the double buffer owns its strings, in the arena of the same index.
    const arena = enricher.arenas[1 - processes.current.load(.acquire)];
    _ = arena.reset(.retain_capacity);

    const back = processes.back();
    back.clearRetainingCapacity();
    for (lists) |list| {
        back.appendSlice(gpa, list.front().items) catch break;
    }

    enricher.enrich(arena.allocator(), io, back.items);
    processes.swap();
}

const ProcessEnricher = struct {
    arenas: [2]*std.heap.ArenaAllocator,
    /// Lives in the arena of the previous poll, which is reset one poll later.
    prev_ticks: std.AutoHashMapUnmanaged(u32, u64) = .empty,
    prev_total_ticks: u64 = 0,

    fn enrich(self: *ProcessEnricher, alloc: std.mem.Allocator, io: std.Io, procs: []ProcessInfo) void {
        const curr_total = readTotalCpuTicks(alloc, io);
        const delta_total = curr_total -| self.prev_total_ticks;

        var curr_ticks: std.AutoHashMapUnmanaged(u32, u64) = .empty;
        curr_ticks.ensureTotalCapacity(alloc, @intCast(procs.len)) catch {};

        for (procs) |*info| {
            if (info.remote) {
                // Already enriched by its host: only the strings are taken over.
                info.username = alloc.dupe(u8, info.username) catch "";
                info.comm = alloc.dupe(u8, info.comm) catch "";
                continue;
            }
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
                    info.cpu_percent = @intCast(@min(delta_proc *| 1000 / delta_total, std.math.maxInt(u16)));
                }
            }

            curr_ticks.put(alloc, info.pid, ticks) catch continue;
        }

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
