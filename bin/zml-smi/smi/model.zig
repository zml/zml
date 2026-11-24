const std = @import("std");

pub const Target = enum { cuda, rocm, oneapi, neuron, tpu };

pub const Device = struct {
    target: Target,
    remote: bool = false,
    name: ?[]const u8 = null,
    driver_version: ?[]const u8 = null,
    details: []const Detail = &.{},

    pub const Detail = struct {
        key: []const u8,
        label: []const u8,
        value: []const u8,
        in_title: bool = false,
    };

    pub const Index = enum(u16) { _ };
};

pub const Sample = union(Target) {
    cuda: Gpu,
    rocm: Gpu,
    oneapi: Gpu,
    neuron: Neuron,
    tpu: Tpu,

    pub fn init(target: Target) Sample {
        return switch (target) {
            inline else => |tag| @unionInit(Sample, @tagName(tag), .{}),
        };
    }
};

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
    pcie_bandwidth_mbps: ?u64 = null,
    pcie_link_gen: ?u64 = null,
    pcie_link_width: ?u64 = null,
};

pub const Neuron = struct {
    util_percent: ?u64 = null,
    mem_used_bytes: ?u64 = null,
    mem_total_bytes: ?u64 = null,
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
};

pub const Platform = struct {
    devices: []const Device,
    two_readings: bool = false,
    vtable: *const VTable,

    pub const VTable = struct {
        poll: *const fn (platform: *Platform, scratch: std.mem.Allocator, samples: []Sample) void,
        /// Called right after `poll`, with the samples it just filled. A row's device is its position in `devices`.
        processes: ?*const fn (platform: *Platform, scratch: std.mem.Allocator, samples: []const Sample, out: *ProcessList) ProcessList.Error!void = null,
        deinit: ?*const fn (platform: *Platform) void = null,
    };
};

pub const ProcessList = struct {
    rows: std.ArrayList(ProcessInfo) = .empty,
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

    pub fn dupe(list: *ProcessList, string: []const u8) error{OutOfMemory}![]const u8 {
        return list.strings.allocator().dupe(u8, string);
    }
};
