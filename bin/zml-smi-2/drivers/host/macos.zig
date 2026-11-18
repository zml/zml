//! The host, from sysctl, the Mach host statistics and libproc.
const std = @import("std");
const contract = @import("zml-smi-2/contract");
const native = @import("zml-smi-2/native");

const sdk = @import("c");

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
        if (c.mach_timebase_info(&timebase) != sdk.KERN_SUCCESS or timebase.denom == 0) timebase = .{ .numer = 1, .denom = 1 };
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
        const loaded = sdk.getloadavg(&load, load.len) == load.len;
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

        if (procInfo(sdk.proc_bsdshortinfo, sdk.PROC_PIDT_SHORTBSDINFO, pid)) |short| {
            row.uid = short.pbsi_uid;
            const username = if (c.getpwuid(row.uid)) |user| user.name else null;
            if (username) |name| row.username = strings.dupe(u8, std.mem.span(name)) catch "";
        }

        var path: [sdk.PROC_PIDPATHINFO_MAXSIZE]u8 = undefined;
        const length = sdk.proc_pidpath(pid, &path, path.len);
        if (length > 0) row.comm = strings.dupe(u8, path[0..@intCast(length)]) catch "";

        const task = procInfo(sdk.proc_taskinfo, sdk.PROC_PIDTASKINFO, pid) orelse return;
        row.rss_kib = task.pti_resident_size / 1024;
        const ticks = @as(u128, task.pti_total_user) + task.pti_total_system;
        const nanoseconds = std.math.lossyCast(u64, ticks * self.timebase.numer / self.timebase.denom);
        const growth = (self.cpu_time.advance(row.pid, self.io, nanoseconds) catch return) orelse return;
        // Tenths of a percent of every CPU, as on Linux.
        const whole = @as(u128, @intCast(growth.elapsed.nanoseconds)) * self.cpus;
        row.cpu_percent = @intCast(@min(@as(u128, growth.amount) * 1000 / whole, std.math.maxInt(u16)));
    }
};

fn procInfo(comptime Info: type, flavor: c_int, pid: c_int) ?Info {
    var info: Info = undefined;
    const size = @sizeOf(Info);
    return if (sdk.proc_pidinfo(pid, flavor, 0, &info, size) == size) info else null;
}

/// Free, inactive and purgeable pages.
fn memAvailable() ?u64 {
    var stats: sdk.vm_statistics64_data_t = undefined;
    var count: sdk.mach_msg_type_number_t = sdk.ZML_HOST_VM_INFO64_COUNT;
    if (sdk.host_statistics64(c.mach_host_self(), sdk.HOST_VM_INFO64, @ptrCast(&stats), &count) != sdk.KERN_SUCCESS) return null;

    var page_size: u64 = 0;
    if (c._host_page_size(c.mach_host_self(), &page_size) != sdk.KERN_SUCCESS) return null;

    const pages = @as(u64, stats.free_count) + stats.inactive_count + stats.purgeable_count;
    return pages * page_size / 1024;
}

fn sysctlString(arena: std.mem.Allocator, name: [*:0]const u8) ?[]const u8 {
    var buffer: [256]u8 = undefined;
    var len: usize = buffer.len;
    if (c.sysctlbyname(name, &buffer, &len, null, 0) != 0) return null;
    return arena.dupe(u8, std.mem.sliceTo(buffer[0..len], 0)) catch null;
}
