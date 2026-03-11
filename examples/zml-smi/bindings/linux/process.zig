const std = @import("std");
const sysfs = @import("../../sysfs.zig");
const worker = @import("../../worker.zig");
const pi = @import("../../info/process_info.zig");

const ProcessInfo = pi.ProcessInfo;
const ProcessList = pi.ProcessList;

const max_samples: usize = 2048;
const scan_interval_ms: u64 = 2000;
const page_size_kib: u64 = 4; // 4096 bytes / 1024

pub const EnrichFn = *const fn ([]ProcessInfo) void;
const max_enrich_fns: usize = 4;

pub const ProcessScanner = struct {
    buffers: [2]ProcessList = .{ .{}, .{} },
    current: std.atomic.Value(u8) = .init(0),
    uid_table: UidTable = .{},
    enrich_fns: [max_enrich_fns]?EnrichFn = .{null} ** max_enrich_fns,
    enrich_count: u8 = 0,

    pub fn addEnrichFn(self: *ProcessScanner, f: EnrichFn) void {
        if (self.enrich_count < max_enrich_fns) {
            self.enrich_fns[self.enrich_count] = f;
            self.enrich_count += 1;
        }
    }

    pub fn getFront(self: *const ProcessScanner) *const ProcessList {
        const idx = self.current.load(.acquire);
        return &self.buffers[idx];
    }

    fn getBack(self: *ProcessScanner) *ProcessList {
        const front_idx = self.current.load(.acquire);
        return &self.buffers[1 - front_idx];
    }

    fn swapBuffers(self: *ProcessScanner) void {
        const front_idx = self.current.load(.acquire);
        self.current.store(1 - front_idx, .release);
    }
};

// UID → username lookup
const max_uid_entries: usize = 512;

const UidEntry = struct {
    uid: u32,
    name: [32]u8,
};

const UidTable = struct {
    entries: [max_uid_entries]UidEntry = undefined,
    count: u32 = 0,

    fn lookup(self: *const UidTable, uid: u32) [32]u8 {
        for (self.entries[0..self.count]) |entry| {
            if (entry.uid == uid) return entry.name;
        }
        // Fallback: format UID as string
        var buf: [32]u8 = .{0} ** 32;
        _ = std.fmt.bufPrint(&buf, "{d}", .{uid}) catch {};
        return buf;
    }
};

fn loadPasswd(io: std.Io, table: *UidTable) void {
    var buf: [32768]u8 = undefined;
    const data = std.Io.Dir.readFile(.cwd(), io, "/etc/passwd", &buf) catch return;
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        // format: name:x:uid:gid:...
        var fields = std.mem.splitScalar(u8, line, ':');
        const name = fields.next() orelse continue;
        _ = fields.next(); // skip password
        const uid_str = fields.next() orelse continue;
        const uid = std.fmt.parseInt(u32, uid_str, 10) catch continue;

        if (table.count >= max_uid_entries) break;
        var entry: UidEntry = .{ .uid = uid, .name = .{0} ** 32 };
        const len = @min(name.len, 31);
        @memcpy(entry.name[0..len], name[0..len]);
        table.entries[table.count] = entry;
        table.count += 1;
    }
}

// Per-process sample for CPU% delta calculation
const ProcSample = struct {
    pid: u32,
    uid: u32,
    ticks: u64, // utime + stime
    rss_pages: u64,
    comm: [256]u8,
    cmdline: [256]u8,
};

pub fn init(io: std.Io, scanner: *ProcessScanner) !void {
    loadPasswd(io, &scanner.uid_table);
    try worker.spawnCustomWorker(io, workerLoop, .{ io, scanner });
}

fn workerLoop(io: std.Io, scanner: *ProcessScanner) void {
    var prev_samples: [max_samples]ProcSample = undefined;
    var prev_count: usize = 0;
    var prev_total_ticks: u64 = 0;
    const interval: std.Io.Duration = .fromMilliseconds(scan_interval_ms);

    // Take initial sample before entering loop
    prev_total_ticks = readTotalCpuTicks(io);
    prev_count = scanProcs(io, &prev_samples);

    while (worker.isRunning()) {
        io.sleep(interval, .awake) catch {};
        if (!worker.isRunning()) break;

        // Take new sample
        var curr_samples: [max_samples]ProcSample = undefined;
        const curr_count = scanProcs(io, &curr_samples);
        const curr_total = readTotalCpuTicks(io);
        const delta_total = curr_total -| prev_total_ticks;

        // Build process list with CPU%
        var result: [max_samples]ProcessInfo = undefined;
        var result_count: usize = 0;

        for (curr_samples[0..curr_count]) |*curr| {
            var info: ProcessInfo = .{
                .pid = curr.pid,
                .uid = curr.uid,
                .username = scanner.uid_table.lookup(curr.uid),
                .rss_kib = curr.rss_pages * page_size_kib,
                .comm = if (hasContent(&curr.cmdline)) curr.cmdline else curr.comm,
            };

            // Find matching previous sample for CPU% delta
            if (delta_total > 0) {
                for (prev_samples[0..prev_count]) |*prev| {
                    if (prev.pid == curr.pid) {
                        const delta_proc = curr.ticks -| prev.ticks;
                        // 100% = all cores busy (btop style)
                        info.cpu_percent = @intCast(@min(delta_proc * 1000 / delta_total, 9999));
                        break;
                    }
                }
            }

            // Skip kernel threads (0 RSS) and idle processes
            if (info.rss_kib == 0 and info.cpu_percent == 0) continue;

            if (result_count < max_samples) {
                result[result_count] = info;
                result_count += 1;
            }
        }

        // Sort by CPU% descending
        std.sort.insertion(ProcessInfo, result[0..result_count], {}, struct {
            fn cmp(_: void, a: ProcessInfo, b: ProcessInfo) bool {
                if (a.cpu_percent != b.cpu_percent) return a.cpu_percent > b.cpu_percent;
                return a.rss_kib > b.rss_kib;
            }
        }.cmp);

        for (scanner.enrich_fns[0..scanner.enrich_count]) |f| f.?(result[0..result_count]);

        // Write top N to back buffer
        const back = scanner.getBack();
        const n = @min(result_count, pi.max_processes);
        back.count = @intCast(n);
        @memcpy(back.entries[0..n], result[0..n]);
        scanner.swapBuffers();

        // Save current as previous for next iteration
        prev_count = curr_count;
        prev_total_ticks = curr_total;
        @memcpy(prev_samples[0..curr_count], curr_samples[0..curr_count]);
    }
}

fn hasContent(buf: *const [256]u8) bool {
    return buf[0] != 0;
}

fn scanProcs(io: std.Io, samples: *[max_samples]ProcSample) usize {
    var count: usize = 0;
    var proc_dir = std.Io.Dir.openDir(.cwd(), io, "/proc", .{ .iterate = true }) catch return 0;
    defer proc_dir.close(io);

    var it = proc_dir.iterate();
    while (it.next(io) catch null) |entry| {
        const pid = std.fmt.parseInt(u32, entry.name, 10) catch continue;
        if (count >= max_samples) break;

        var path_buf: [64]u8 = undefined;

        // Read /proc/[pid]/stat
        const stat_path = std.fmt.bufPrint(&path_buf, "/proc/{d}/stat", .{pid}) catch continue;
        var stat_buf: [1024]u8 = undefined;
        const stat_data = std.Io.Dir.readFile(.cwd(), io, stat_path, &stat_buf) catch continue;
        const parsed = parseStat(stat_data) orelse continue;

        // Read UID from /proc/[pid]/status
        var status_path_buf: [64]u8 = undefined;
        const status_path = std.fmt.bufPrint(&status_path_buf, "/proc/{d}/status", .{pid}) catch continue;
        const uid: u32 = @intCast(sysfs.readFieldInt(io, status_path, "Uid") catch 0);

        // Read cmdline
        var cmdline_path_buf: [64]u8 = undefined;
        const cmdline = readCmdline(io, &cmdline_path_buf, pid);

        samples[count] = .{
            .pid = pid,
            .uid = uid,
            .ticks = parsed.utime + parsed.stime,
            .rss_pages = parsed.rss,
            .comm = parsed.comm,
            .cmdline = cmdline,
        };
        count += 1;
    }
    return count;
}

const StatFields = struct {
    utime: u64,
    stime: u64,
    rss: u64,
    comm: [256]u8,
};

fn parseStat(data: []const u8) ?StatFields {
    // Find comm field boundaries: (comm_name)
    const comm_start = (std.mem.indexOfScalar(u8, data, '(') orelse return null) + 1;
    const comm_end = std.mem.lastIndexOfScalar(u8, data, ')') orelse return null;

    var comm_buf: [256]u8 = .{0} ** 256;
    const comm_len = @min(comm_end - comm_start, 255);
    @memcpy(comm_buf[0..comm_len], data[comm_start..][0..comm_len]);

    // Fields after ") " are 0-indexed: state=0, ppid=1, ..., utime=11, stime=12, ..., rss=21
    if (comm_end + 2 >= data.len) return null;
    const rest = data[comm_end + 2 ..];
    var field_idx: usize = 0;
    var utime: u64 = 0;
    var stime: u64 = 0;
    var rss: u64 = 0;

    var iter = std.mem.splitScalar(u8, rest, ' ');
    while (iter.next()) |field| {
        switch (field_idx) {
            11 => utime = std.fmt.parseInt(u64, field, 10) catch 0,
            12 => stime = std.fmt.parseInt(u64, field, 10) catch 0,
            21 => rss = std.fmt.parseInt(u64, field, 10) catch 0,
            else => {},
        }
        field_idx += 1;
        if (field_idx > 21) break;
    }

    return .{ .utime = utime, .stime = stime, .rss = rss, .comm = comm_buf };
}

fn readCmdline(io: std.Io, path_buf: *[64]u8, pid: u32) [256]u8 {
    const path = std.fmt.bufPrint(path_buf, "/proc/{d}/cmdline", .{pid}) catch return .{0} ** 256;
    var buf: [256]u8 = .{0} ** 256;
    const data = std.Io.Dir.readFile(.cwd(), io, path, &buf) catch return .{0} ** 256;
    // Replace null bytes with spaces, trim trailing
    const len = std.mem.trimEnd(u8, data, &.{0}).len;
    for (buf[0..len]) |*c| {
        if (c.* == 0) c.* = ' ';
    }
    // Zero out rest
    if (len < 256) @memset(buf[len..], 0);
    return buf;
}

fn readTotalCpuTicks(io: std.Io) u64 {
    var buf: [512]u8 = undefined;
    const data = std.Io.Dir.readFile(.cwd(), io, "/proc/stat", &buf) catch return 0;
    const first_line_end = std.mem.indexOfScalar(u8, data, '\n') orelse data.len;
    const line = data[0..first_line_end];

    var total: u64 = 0;
    var iter = std.mem.splitScalar(u8, line, ' ');
    _ = iter.next(); // skip "cpu"
    while (iter.next()) |field| {
        if (field.len == 0) continue;
        total += std.fmt.parseInt(u64, field, 10) catch continue;
    }
    return total;
}
