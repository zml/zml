const std = @import("std");

pub const MonitorReport = struct {
    neuron_runtime_data: []const RuntimeData = &.{},
    neuron_hardware_info: HardwareInfo = .{},
};

pub const HardwareInfo = struct {
    neuron_device_memory_size: u64 = 0,
    neuroncore_per_device_count: u64 = 0,
};

pub const RuntimeData = struct {
    pid: u32 = 0,
    report: Report = .{},
};

pub const Report = struct {
    neuroncore_counters: ?NeuroncoreCounters = null,
    memory_used: ?MemoryUsed = null,
};

pub const MemoryUsed = struct {
    neuron_runtime_used_bytes: RuntimeUsedBytes = .{},
};

pub const RuntimeUsedBytes = struct {
    neuron_device: u64 = 0,
    usage_breakdown: ?UsageBreakdown = null,
};

pub const UsageBreakdown = struct {
    neuroncore_memory_usage: std.json.ArrayHashMap(NeuroncoreMemUsage) = .{},
};

pub const NeuroncoreMemUsage = struct {
    constants: u64 = 0,
    model_code: u64 = 0,
    model_shared_scratchpad: u64 = 0,
    runtime_memory: u64 = 0,
    tensors: u64 = 0,
};

pub const NeuroncoreCounters = struct {
    neuroncores_in_use: std.json.ArrayHashMap(NeuroncoreUsage) = .{},
};

pub const NeuroncoreUsage = struct {
    neuroncore_utilization: f64 = 0,
};
