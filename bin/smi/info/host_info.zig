const DoubleBuffer = @import("../utils/double_buffer.zig").DoubleBuffer;

pub const HostInfo = DoubleBuffer(HostData);

pub const HostData = struct {
    hostname: ?[256]u8 = null,
    kernel: ?[256]u8 = null,
    cpu_name: ?[256]u8 = null,
    cpu_cores: ?u64 = null,
    mem_total_kib: ?u64 = null,
    mem_available_kib: ?u64 = null,
    load_avg: ?[256]u8 = null,
    uptime_seconds: ?u64 = null,
};
