const std = @import("std");
const DoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer;

pub const Target = enum {
    cuda,
    rocm,
    neuron,
    tpu,
};

pub const DeviceInfo = union(Target) {
    cuda: DoubleBuffer(GpuInfo),
    rocm: DoubleBuffer(GpuInfo),
    neuron: DoubleBuffer(NeuronInfo),
    tpu: DoubleBuffer(TpuInfo),
};

pub const GpuInfo = struct {
    name: ?[]const u8 = null,

    // Utilization
    util_percent: ?u64 = null,
    encoder_util_percent: ?u64 = null,
    decoder_util_percent: ?u64 = null,

    // Power
    power_mw: ?u64 = null,
    power_limit_mw: ?u64 = null,
    // Thermal
    temperature: ?u64 = null,
    fan_speed_percent: ?u64 = null,

    // Clocks
    clock_graphics_mhz: ?u64 = null,
    clock_sm_mhz: ?u64 = null,
    clock_soc_mhz: ?u64 = null,
    clock_mem_mhz: ?u64 = null,
    clock_graphics_max_mhz: ?u64 = null,
    clock_mem_max_mhz: ?u64 = null,

    // Memory
    mem_used_bytes: ?u64 = null,
    mem_total_bytes: ?u64 = null,
    mem_bus_width: ?u64 = null,

    // PCIe
    pcie_tx_kbps: ?u64 = null,
    pcie_rx_kbps: ?u64 = null,
    pcie_bandwidth_mbps: ?u64 = null,
    pcie_link_gen: ?u64 = null,
    pcie_link_width: ?u64 = null,
};

pub const NeuronInfo = struct {
    name: ?[]const u8 = null,

    util_percent: ?u64 = null,
    mem_used_bytes: ?u64 = null,
    mem_total_bytes: ?u64 = null,

    // Utilization delta tracking (written by metrics worker)
    prev_total_us: u64 = 0,
    prev_timestamp: ?std.Io.Timestamp = null,

    // Neuron per-core HBM breakdown (bytes)
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

pub const TpuInfo = struct {
    name: ?[]const u8 = null,

    util_percent: ?u64 = null,
    mem_used_bytes: ?u64 = null,
    mem_total_bytes: ?u64 = null,
};
