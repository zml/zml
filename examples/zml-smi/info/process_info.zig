pub const max_processes: usize = 50;

pub const ProcessInfo = struct {
    pid: u32 = 0,
    uid: u32 = 0,
    username: [32]u8 = .{0} ** 32,
    comm: [256]u8 = .{0} ** 256,
    cpu_percent: u16 = 0, // *10, e.g. 125 = 12.5%
    rss_kib: u64 = 0,
    // GPU fields — enriched by backends
    device_idx: ?u8 = null,
    gpu_util_percent: ?u16 = null,
    gpu_mem_kib: ?u64 = null,
};

pub const ProcessList = struct {
    entries: [max_processes]ProcessInfo = [_]ProcessInfo{.{}} ** max_processes,
    count: u32 = 0,
};
