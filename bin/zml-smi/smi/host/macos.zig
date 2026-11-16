//! The host, from sysctl, the Mach host statistics and libproc.
const std = @import("std");
const smi = @import("../smi.zig");

const c = std.c;

pub const Host = struct {
    io: std.Io,
    /// CPU nanoseconds of each process.
    cpu_time: smi.Counters(u32),
    cpus: u64,
    /// Converts the CPU time libproc reports, in Mach ticks, to nanoseconds.
    timebase: c.mach_timebase_info_data,

    /// Reads what does not change: hostname, kernel, CPU, core count and total memory.
    pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, data: *smi.HostData) Host {
        data.hostname = sysctlString(arena, "kern.hostname");
        data.kernel = sysctlString(arena, "kern.osrelease");
        data.cpu_name = sysctlString(arena, "machdep.cpu.brand_string");
        data.cpu_cores = if (sysctl(u32, "hw.logicalcpu")) |cores| cores else null;
        data.mem_total_kib = if (sysctl(u64, "hw.memsize")) |bytes| bytes / 1024 else null;

        var timebase: c.mach_timebase_info_data = .{ .numer = 1, .denom = 1 };
        if (c.mach_timebase_info(&timebase) != KERN_SUCCESS or timebase.denom == 0) timebase = .{ .numer = 1, .denom = 1 };
        return .{ .io = io, .cpu_time = .init(gpa), .cpus = @max(1, data.cpu_cores orelse 1), .timebase = timebase };
    }

    pub fn deinit(self: *Host) void {
        self.cpu_time.deinit();
    }

    /// Overwrites what changes: available memory, load and uptime. A read that fails gives null.
    pub fn sample(_: *Host, data: *smi.HostData) void {
        data.mem_available_kib = memAvailable();

        var load: [3]f64 = undefined;
        const loaded = getloadavg(&load, 3) == 3;
        inline for (.{ &data.load_1, &data.load_5, &data.load_15 }, 0..) |field, i| {
            field.* = if (loaded) @floatCast(load[i]) else null;
        }

        data.uptime_seconds = null;
        var now: c.timeval = undefined;
        if (sysctl(c.timeval, "kern.boottime")) |boot| {
            if (c.gettimeofday(&now, null) == 0) data.uptime_seconds = @intCast(now.sec - boot.sec);
        }
    }

    /// Adds user, command, CPU and RSS to one row per local PID.
    /// The CPU and RSS of another user's process read 0, unless zml-smi runs as root.
    pub fn enrich(self: *Host, strings: std.mem.Allocator, rows: []smi.ProcessInfo) void {
        defer self.cpu_time.sweep();

        for (rows) |*row| self.read(strings, row);
    }

    fn read(self: *Host, strings: std.mem.Allocator, row: *smi.ProcessInfo) void {
        const pid = std.math.cast(c_int, row.pid) orelse return;

        var short: ProcBsdShortInfo = undefined;
        if (proc_pidinfo(pid, PROC_PIDT_SHORTBSDINFO, 0, &short, @sizeOf(ProcBsdShortInfo)) == @sizeOf(ProcBsdShortInfo)) {
            row.uid = short.uid;
            if (c.getpwuid(row.uid)) |user| {
                if (user.name) |name| row.username = strings.dupe(u8, std.mem.span(name)) catch "";
            }
        }

        var path: [PROC_PIDPATHINFO_MAXSIZE]u8 = undefined;
        const length = proc_pidpath(pid, &path, path.len);
        if (length > 0) row.comm = strings.dupe(u8, path[0..@intCast(length)]) catch "";

        var task: ProcTaskInfo = undefined;
        if (proc_pidinfo(pid, PROC_PIDTASKINFO, 0, &task, @sizeOf(ProcTaskInfo)) != @sizeOf(ProcTaskInfo)) return;
        row.rss_kib = task.resident_size / 1024;
        const ticks = @as(u128, task.total_user) + task.total_system;
        const nanoseconds = std.math.lossyCast(u64, ticks * self.timebase.numer / self.timebase.denom);
        const growth = (self.cpu_time.advance(row.pid, self.io, nanoseconds) catch return) orelse return;
        // Tenths of a percent of every CPU, as on Linux.
        const whole = @as(u128, @intCast(growth.elapsed.nanoseconds)) * self.cpus;
        row.cpu_percent = @intCast(@min(@as(u128, growth.amount) * 1000 / whole, std.math.maxInt(u16)));
    }
};

const PROC_PIDTASKINFO: c_int = 4;
const PROC_PIDT_SHORTBSDINFO: c_int = 13;
const PROC_PIDPATHINFO_MAXSIZE = 4096;

/// struct proc_bsdshortinfo, which libproc gives for every process.
const ProcBsdShortInfo = extern struct {
    pid: u32,
    ppid: u32,
    pgid: u32,
    status: u32,
    comm: [16]u8,
    flags: u32,
    uid: c.uid_t,
    gid: c.gid_t,
    ruid: c.uid_t,
    rgid: c.gid_t,
    svuid: c.uid_t,
    svgid: c.gid_t,
    rfu: u32,
};

/// struct proc_taskinfo, which libproc gives for the user's own processes, or any as root.
const ProcTaskInfo = extern struct {
    virtual_size: u64,
    resident_size: u64,
    total_user: u64,
    total_system: u64,
    threads_user: u64,
    threads_system: u64,
    policy: i32,
    faults: i32,
    pageins: i32,
    cow_faults: i32,
    messages_sent: i32,
    messages_received: i32,
    syscalls_mach: i32,
    syscalls_unix: i32,
    csw: i32,
    threadnum: i32,
    numrunning: i32,
    priority: i32,
};

extern "c" fn proc_pidinfo(pid: c_int, flavor: c_int, arg: u64, buffer: *anyopaque, size: c_int) c_int;
extern "c" fn proc_pidpath(pid: c_int, buffer: [*]u8, size: u32) c_int;

const vm_statistics64 = extern struct {
    free_count: c_uint,
    active_count: c_uint,
    inactive_count: c_uint,
    wire_count: c_uint,
    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,
    purgeable_count: c_uint,
    speculative_count: c_uint,
    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,
    compressor_page_count: c_uint,
    throttled_count: c_uint,
    external_page_count: c_uint,
    internal_page_count: c_uint,
    total_uncompressed_pages_in_compressor: u64,
};

const HOST_VM_INFO64: c_int = 4;
const KERN_SUCCESS: c_int = 0;

extern "c" fn host_statistics64(host: c.mach_port_t, flavor: c_int, info: *anyopaque, count: *c_uint) c.kern_return_t;
extern "c" fn getloadavg(loadavg: *[3]f64, nelem: c_int) c_int;

/// Free, inactive and purgeable pages.
fn memAvailable() ?u64 {
    var stats: vm_statistics64 = undefined;
    var count: c_uint = @sizeOf(vm_statistics64) / @sizeOf(c_int);
    if (host_statistics64(c.mach_host_self(), HOST_VM_INFO64, &stats, &count) != KERN_SUCCESS) return null;

    var page_size: u64 = 0;
    if (c._host_page_size(c.mach_host_self(), &page_size) != KERN_SUCCESS) return null;

    const pages = @as(u64, stats.free_count) + stats.inactive_count + stats.purgeable_count;
    return pages * page_size / 1024;
}

fn sysctl(comptime T: type, name: [*:0]const u8) ?T {
    var value: T = undefined;
    var len: usize = @sizeOf(T);
    if (c.sysctlbyname(name, std.mem.asBytes(&value), &len, null, 0) != 0) return null;
    return value;
}

fn sysctlString(arena: std.mem.Allocator, name: [*:0]const u8) ?[]const u8 {
    var buffer: [256]u8 = undefined;
    var len: usize = buffer.len;
    if (c.sysctlbyname(name, &buffer, &len, null, 0) != 0) return null;
    return arena.dupe(u8, std.mem.sliceTo(buffer[0..len], 0)) catch null;
}
