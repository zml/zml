//! The host, from sysctl, the Mach host statistics and libproc.
const std = @import("std");
const contract = @import("zml-smi-2/contract");
const native = @import("zml-smi-2/native");

const c = std.c;

pub const Host = struct {
    io: std.Io,
    /// CPU nanoseconds of each process.
    cpu_time: native.Counters(u32),
    cpus: u64,
    /// Converts the CPU time libproc reports, in Mach ticks, to nanoseconds.
    timebase: c.mach_timebase_info_data,

    /// Reads what does not change: hostname, kernel, CPU, core count and total memory.
    pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, data: *contract.HostData) Host {
        data.hostname = sysctlString(arena, "kern.hostname");
        data.kernel = sysctlString(arena, "kern.osrelease");
        data.cpu_name = sysctlString(arena, "machdep.cpu.brand_string");
        data.cpu_cores = if (native.sysctl(u32, "hw.logicalcpu")) |cores| cores else null;
        data.mem_total_kib = if (native.sysctl(u64, "hw.memsize")) |bytes| bytes / 1024 else null;

        var timebase: c.mach_timebase_info_data = .{ .numer = 1, .denom = 1 };
        if (c.mach_timebase_info(&timebase) != KERN_SUCCESS or timebase.denom == 0) timebase = .{ .numer = 1, .denom = 1 };
        return .{ .io = io, .cpu_time = .init(gpa), .cpus = @max(1, data.cpu_cores orelse 1), .timebase = timebase };
    }

    pub fn deinit(self: *Host) void {
        self.cpu_time.deinit();
    }

    /// Refreshes dynamic host metrics and enriches one row per local PID.
    /// Another user's process only exposes CPU and RSS when the monitor runs as root.
    pub fn read(
        self: *Host,
        strings: std.mem.Allocator,
        rows: []contract.ProcessInfo,
        data: *contract.HostData,
    ) void {
        for (rows) |*row| self.readProcess(strings, row);
        self.cpu_time.sweep();
        data.mem_available_kib = memAvailable();

        var load: [3]f64 = undefined;
        const loaded = getloadavg(&load, load.len) == load.len;
        for ([_]*?f32{ &data.load_1, &data.load_5, &data.load_15 }, 0..) |field, index| {
            field.* = if (loaded) @floatCast(load[index]) else null;
        }

        data.uptime_seconds = null;
        var now: c.timeval = undefined;
        if (native.sysctl(c.timeval, "kern.boottime")) |boot| {
            if (c.gettimeofday(&now, null) == 0) data.uptime_seconds = @intCast(now.sec - boot.sec);
        }
    }

    fn readProcess(self: *Host, strings: std.mem.Allocator, row: *contract.ProcessInfo) void {
        const pid = std.math.cast(c_int, row.pid) orelse return;

        if (procInfo(ProcBsdShortInfo, pid)) |short| {
            row.uid = short.uid;
            const username = if (c.getpwuid(row.uid)) |user| user.name else null;
            if (username) |name| row.username = strings.dupe(u8, std.mem.span(name)) catch "";
        }

        var path: [PROC_PIDPATHINFO_MAXSIZE]u8 = undefined;
        const length = proc_pidpath(pid, &path, path.len);
        if (length > 0) row.comm = strings.dupe(u8, path[0..@intCast(length)]) catch "";

        const task = procInfo(ProcTaskInfo, pid) orelse return;
        row.rss_kib = task.resident_size / 1024;
        const ticks = @as(u128, task.total_user) + task.total_system;
        const nanoseconds = std.math.lossyCast(u64, ticks * self.timebase.numer / self.timebase.denom);
        const growth = (self.cpu_time.advance(row.pid, self.io, nanoseconds) catch return) orelse return;
        // Tenths of a percent of every CPU, as on Linux.
        const whole = @as(u128, @intCast(growth.elapsed.nanoseconds)) * self.cpus;
        row.cpu_percent = @intCast(@min(@as(u128, growth.amount) * 1000 / whole, std.math.maxInt(u16)));
    }
};

const PROC_PIDPATHINFO_MAXSIZE = 4096;

/// struct proc_bsdshortinfo, which libproc gives for every process.
/// Unused members are grouped below. The layout comes from <sys/proc_info.h>.
const ProcBsdShortInfo = extern struct {
    ids_status_comm_and_flags: [9]u32,
    uid: c.uid_t,
    groups_saved_ids_and_reserved: [6]u32,

    const flavor = 13;
};

/// struct proc_taskinfo, which libproc gives for the user's own processes, or any as root.
const ProcTaskInfo = extern struct {
    virtual_size: u64,
    resident_size: u64,
    total_user: u64,
    total_system: u64,
    thread_times: [2]u64,
    policy_and_activity: [12]i32,

    const flavor = 4;
};

fn procInfo(comptime Info: type, pid: c_int) ?Info {
    var info: Info = undefined;
    const size = @sizeOf(Info);
    return if (proc_pidinfo(pid, Info.flavor, 0, &info, size) == size) info else null;
}

extern "c" fn proc_pidinfo(pid: c_int, flavor: c_int, arg: u64, buffer: *anyopaque, size: c_int) c_int;
extern "c" fn proc_pidpath(pid: c_int, buffer: [*]u8, size: u32) c_int;

/// Revision 1 of struct vm_statistics64 from <mach/vm_statistics.h>.
const VmStatistics64 = extern struct {
    free_count: c_uint,
    active_count: c_uint,
    inactive_count: c_uint,
    wire_count: c_uint,
    activity: [9]u64,
    purgeable_count: c_uint,
    speculative_and_compression_stats: [15]c_uint,
};

comptime {
    std.debug.assert(@sizeOf(ProcBsdShortInfo) == 64 and @offsetOf(ProcBsdShortInfo, "uid") == 36);
    std.debug.assert(@sizeOf(ProcTaskInfo) == 96 and @offsetOf(ProcTaskInfo, "resident_size") == 8);
    std.debug.assert(@offsetOf(ProcTaskInfo, "total_user") == 16);
    std.debug.assert(@offsetOf(ProcTaskInfo, "total_system") == 24);
    std.debug.assert(@sizeOf(VmStatistics64) == 152 and @offsetOf(VmStatistics64, "free_count") == 0);
    std.debug.assert(@offsetOf(VmStatistics64, "inactive_count") == 8);
    std.debug.assert(@offsetOf(VmStatistics64, "purgeable_count") == 88);
}

const HOST_VM_INFO64: c_int = 4;
const KERN_SUCCESS: c_int = 0;

extern "c" fn host_statistics64(host: c.mach_port_t, flavor: c_int, info: *anyopaque, count: *c_uint) c.kern_return_t;
extern "c" fn getloadavg(loadavg: *[3]f64, nelem: c_int) c_int;

/// Free, inactive and purgeable pages.
fn memAvailable() ?u64 {
    var stats: VmStatistics64 = undefined;
    var count: c_uint = @sizeOf(VmStatistics64) / @sizeOf(c_int);
    if (host_statistics64(c.mach_host_self(), HOST_VM_INFO64, &stats, &count) != KERN_SUCCESS) return null;

    var page_size: u64 = 0;
    if (c._host_page_size(c.mach_host_self(), &page_size) != KERN_SUCCESS) return null;

    const pages = @as(u64, stats.free_count) + stats.inactive_count + stats.purgeable_count;
    return pages * page_size / 1024;
}

fn sysctlString(arena: std.mem.Allocator, name: [*:0]const u8) ?[]const u8 {
    var buffer: [256]u8 = undefined;
    var len: usize = buffer.len;
    if (c.sysctlbyname(name, &buffer, &len, null, 0) != 0) return null;
    return arena.dupe(u8, std.mem.sliceTo(buffer[0..len], 0)) catch null;
}
