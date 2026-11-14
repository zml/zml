const std = @import("std");

pub const Target = enum { cuda, rocm, oneapi, neuron, tpu, furiosa, metal };

/// What does not change while we run. Its strings live as long as the program.
pub const Device = struct {
    target: Target,
    /// It comes from another zml-smi, over --remotes.
    remote: bool = false,
    name: ?[]const u8 = null,
    driver_version: ?[]const u8 = null,
    /// The strings only one platform has, such as the CUDA version.
    details: []const Detail = &.{},

    pub const Detail = struct {
        /// Its name in JSON, CSV and Prometheus labels: "cuda_version".
        key: []const u8,
        /// Its name in the TUI: "CUDA".
        label: []const u8,
        value: []const u8,
        /// Shown beside the driver version in the card title.
        in_title: bool = false,
    };

    /// Devices are referred to by index, never by pointer.
    pub const Index = enum(u16) { _ };
};

/// What is read every poll. Numbers only. Its tag is the device's target.
pub const Sample = union(Target) {
    cuda: Gpu,
    rocm: Gpu,
    oneapi: Gpu,
    neuron: Neuron,
    tpu: Tpu,
    furiosa: Furiosa,
    metal: Metal,

    /// A sample of the given target with every field null.
    pub fn init(target: Target) Sample {
        return switch (target) {
            inline else => |tag| @unionInit(Sample, @tagName(tag), .{}),
        };
    }
};

/// A device and its latest sample: what JSON writes and what a remote host sends.
pub const Row = struct { device: Device, sample: Sample };

/// Every GPU target. CSV columns and JSON keys follow the field order.
pub const Gpu = struct {
    util_percent: ?u64 = null,
    encoder_util_percent: ?u64 = null,
    decoder_util_percent: ?u64 = null,
    power_mw: ?u64 = null,
    power_limit_mw: ?u64 = null,
    /// Degrees Celsius.
    temperature: ?u64 = null,
    fan_speed_percent: ?u64 = null,
    clock_graphics_mhz: ?u64 = null,
    clock_sm_mhz: ?u64 = null,
    clock_soc_mhz: ?u64 = null,
    clock_mem_mhz: ?u64 = null,
    clock_graphics_max_mhz: ?u64 = null,
    clock_mem_max_mhz: ?u64 = null,
    mem_used_bytes: ?u64 = null,
    mem_total_bytes: ?u64 = null,
    /// Bits.
    mem_bus_width: ?u64 = null,
    pcie_tx_kbps: ?u64 = null,
    pcie_rx_kbps: ?u64 = null,
    /// The line rate of the negotiated link across all its lanes.
    pcie_bandwidth_mbps: ?u64 = null,
    pcie_link_gen: ?u64 = null,
    pcie_link_width: ?u64 = null,
};

/// One Neuron core.
pub const Neuron = struct {
    util_percent: ?u64 = null,
    mem_used_bytes: ?u64 = null,
    mem_total_bytes: ?u64 = null,
    // Device memory of the core by category, in bytes.
    nc_tensors: ?u64 = null,
    nc_constants: ?u64 = null,
    nc_model_code: ?u64 = null,
    nc_shared_scratchpad: ?u64 = null,
    nc_nonshared_scratchpad: ?u64 = null,
    nc_runtime: ?u64 = null,
    nc_driver: ?u64 = null,
    nc_dma_rings: ?u64 = null,
    nc_collectives: ?u64 = null,
    nc_notifications: ?u64 = null,
    nc_uncategorized: ?u64 = null,
};

pub const Tpu = struct {
    util_percent: ?u64 = null,
    mem_used_bytes: ?u64 = null,
    mem_total_bytes: ?u64 = null,
};

/// One FuriosaAI RNGD chip.
pub const Furiosa = struct {
    util_percent: ?u64 = null,
    mem_used_bytes: ?u64 = null,
    mem_total_bytes: ?u64 = null,
    power_mw: ?u64 = null,
    power_limit_mw: ?u64 = null,
    /// Degrees Celsius, at the hottest point of the SoC.
    temperature: ?u64 = null,
    /// Degrees Celsius.
    ambient_temperature: ?u64 = null,
    /// A `Governor`.
    governor: ?u64 = null,
    /// A `Liveness`.
    liveness: ?u64 = null,
    /// A bitmask whose bits are `ThrottleReason`s.
    throttle_reasons: ?u64 = null,

    pub const Governor = enum(u64) { performance, powersave };
    pub const Liveness = enum(u64) { unknown, alive };
    pub const ThrottleReason = enum(u6) { idle, thermal_slowdown, app_power_cap, app_clock_cap, hw_clock_cap, hw_bus_limit, hw_power_cap, gov_clock_cap, other };
};

/// An Apple GPU. Its memory is the host's.
pub const Metal = struct {
    util_percent: ?u64 = null,
    mem_used_bytes: ?u64 = null,
    mem_total_bytes: ?u64 = null,
    power_mw: ?u64 = null,
    /// Degrees Celsius, of the hottest GPU sensor.
    temperature: ?u64 = null,
    /// The fastest fan.
    fan_speed_percent: ?u64 = null,
    clock_graphics_mhz: ?u64 = null,
    clock_graphics_max_mhz: ?u64 = null,
    /// The Neural Engine.
    ane_power_mw: ?u64 = null,
};

pub const ProcessInfo = struct {
    pid: u32 = 0,
    /// The global index of the device, once the collector has the row.
    device_idx: u16 = 0,
    dev_util_percent: ?u16 = null,
    dev_mem_kib: ?u64 = null,
    uid: u32 = 0,
    username: []const u8 = "",
    comm: []const u8 = "",
    /// Tenths of a percent of every CPU.
    cpu_percent: u16 = 0,
    rss_kib: u64 = 0,
    remote: bool = false,
};

pub const HostData = struct {
    hostname: ?[]const u8 = null,
    kernel: ?[]const u8 = null,
    cpu_name: ?[]const u8 = null,
    cpu_cores: ?u64 = null,
    mem_total_kib: ?u64 = null,
    mem_available_kib: ?u64 = null,
    load_1: ?f32 = null,
    load_5: ?f32 = null,
    load_15: ?f32 = null,
    uptime_seconds: ?u64 = null,
    /// Measured by a platform, through `Platform.VTable.host`.
    cpu_power_mw: ?u64 = null,
    system_power_mw: ?u64 = null,
};

/// A source of devices: a local platform or a remote host.
/// It is a field of its implementation, which finds itself with @fieldParentPtr.
pub const Platform = struct {
    /// The devices found when the platform was opened. The list never changes afterwards.
    devices: []const Device,
    /// Values are derived from two polls, so a one-shot print polls twice.
    two_readings: bool = false,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Called every poll, with one empty sample per device, in the order of `devices`.
        poll: *const fn (platform: *Platform, scratch: std.mem.Allocator, samples: []Sample) void,
        /// Called right after `poll`, with the samples it just filled. A row's device is its position in `devices`.
        processes: ?*const fn (platform: *Platform, scratch: std.mem.Allocator, samples: []const Sample, out: *ProcessList) ProcessList.Error!void = null,
        deinit: ?*const fn (platform: *Platform) void = null,
        /// For a platform that also measures the host. Called right after `processes`, in the same task,
        /// with host data whose fields start null: it fills those it read.
        host: ?*const fn (platform: *Platform, data: *HostData) void = null,
    };
};

/// The process rows of one source for one poll, and the strings they point into.
pub const ProcessList = struct {
    rows: std.ArrayList(ProcessInfo) = .empty,
    /// Owns every string a row holds.
    strings: std.heap.ArenaAllocator,
    gpa: std.mem.Allocator,

    pub const Error = error{ OutOfMemory, Unavailable };

    pub fn init(gpa: std.mem.Allocator) ProcessList {
        return .{ .strings = .init(gpa), .gpa = gpa };
    }

    pub fn deinit(list: *ProcessList) void {
        list.rows.deinit(list.gpa);
        list.strings.deinit();
    }

    pub fn clear(list: *ProcessList) void {
        list.rows.clearRetainingCapacity();
        _ = list.strings.reset(.retain_capacity);
    }

    pub fn append(list: *ProcessList, row: ProcessInfo) error{OutOfMemory}!void {
        try list.rows.append(list.gpa, row);
    }

    /// Copies a string into the list, for a row to hold.
    pub fn dupe(list: *ProcessList, string: []const u8) error{OutOfMemory}![]const u8 {
        return list.strings.allocator().dupe(u8, string);
    }
};
