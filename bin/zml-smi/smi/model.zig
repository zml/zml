const std = @import("std");
const telemetry = @import("telemetry.zig");

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

    /// The only platform projection needed by shared presentation and history.
    pub fn common(self: Sample) Common {
        var result: Common = .{};
        switch (self) {
            inline else => |values| {
                inline for (@typeInfo(Common).@"struct".field_names) |field| {
                    if (@hasField(@TypeOf(values), field)) @field(result, field) = @field(values, field);
                }
            },
        }
        return result;
    }
};

/// Values shared by cards and history; missing platform fields remain null.
pub const Common = struct {
    util_percent: ?u64 = null,
    mem_used_bytes: ?u64 = null,
    mem_total_bytes: ?u64 = null,
    temperature: ?u64 = null,
    fan_speed_percent: ?u64 = null,
    power_mw: ?u64 = null,
    power_limit_mw: ?u64 = null,
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

    pub const poll = telemetry.poll;
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

    pub const poll = telemetry.poll;
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

    pub const poll = telemetry.poll;

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

    pub const poll = telemetry.poll;
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
    /// Measured alongside the device samples by sources such as Metal.
    cpu_power_mw: ?u64 = null,
    system_power_mw: ?u64 = null,
};
