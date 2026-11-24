const std = @import("std");
const smi = @import("../smi.zig");

const c = std.c;

pub const Host = struct {
    pub fn init(arena: std.mem.Allocator, _: std.mem.Allocator, _: std.Io, data: *smi.HostData) Host {
        data.hostname = sysctlString(arena, "kern.hostname");
        data.kernel = sysctlString(arena, "kern.osrelease");
        data.cpu_name = sysctlString(arena, "machdep.cpu.brand_string");
        data.cpu_cores = if (sysctl(u32, "hw.logicalcpu")) |cores| cores else null;
        data.mem_total_kib = if (sysctl(u64, "hw.memsize")) |bytes| bytes / 1024 else null;
        return .{};
    }

    pub fn deinit(_: *Host) void {}

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
            if (c.gettimeofday(&now, null) == 0) {
                data.uptime_seconds = if (std.math.sub(c.time_t, now.sec, boot.sec)) |diff|
                    std.math.cast(u64, diff)
                else |_|
                    null;
            }
        }
    }

    pub fn enrich(_: *Host, _: std.mem.Allocator, _: std.mem.Allocator, _: []smi.ProcessInfo) void {}
};

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

fn memAvailable() ?u64 {
    const host = c.mach_host_self();
    defer _ = c.mach_port_deallocate(c.mach_task_self(), host);

    var stats: vm_statistics64 = undefined;
    var count: c_uint = @sizeOf(vm_statistics64) / @sizeOf(c_int);
    if (host_statistics64(host, HOST_VM_INFO64, &stats, &count) != KERN_SUCCESS) return null;

    var page_size: u64 = 0;
    if (c._host_page_size(host, &page_size) != KERN_SUCCESS) return null;

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
