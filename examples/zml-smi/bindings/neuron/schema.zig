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
    report: Report = .{},
};

pub const Report = struct {
    neuroncore_counters: ?NeuroncoreCounters = null,
};

pub const NeuroncoreCounters = struct {
    neuroncores_in_use: std.json.ArrayHashMap(NeuroncoreUsage) = .{},
};

pub const NeuroncoreUsage = struct {
    neuroncore_utilization: f64 = 0,
};
