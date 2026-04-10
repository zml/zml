pub const ProcessInfo = struct {
    pid: u32 = 0,
    device_idx: u8 = 0,
    dev_util_percent: ?u16 = null,
    dev_mem_kib: ?u64 = null,
    uid: u32 = 0,
    username: []const u8 = "",
    comm: []const u8 = "",
    cpu_percent: u16 = 0, // *10, e.g. 125 = 12.5%
    rss_kib: u64 = 0,
};
