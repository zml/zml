const std = @import("std");

pub const ProcessInfo = struct {
    pid: u32 = 0,
    device_idx: u8 = 0,
    dev_util_percent: ?u16 = null,
    dev_mem_kib: ?u64 = null,
    uid: u32 = 0,
    username: [32]u8 = .{0} ** 32,
    comm: ?[std.Io.Dir.max_path_bytes]u8 = null,
    cpu_percent: u16 = 0, // *10, e.g. 125 = 12.5%
    rss_kib: u64 = 0,
};
