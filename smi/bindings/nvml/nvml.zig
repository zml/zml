const std = @import("std");
const c = @import("c");
const DynLib = @import("../dynlib.zig").Lib(c);

pub const Error = ReturnError || error{NvmlUnavailable};
pub const Handle = c.nvmlDevice_t;
pub const ProcessInfo_t = c.nvmlProcessInfo_t;

pub const ReturnError = error{
    error_uninitialized,
    error_invalid_argument,
    error_not_supported,
    error_no_permission,
    error_already_initialized,
    error_not_found,
    error_insufficient_size,
    error_insufficient_power,
    error_driver_not_loaded,
    error_timeout,
    error_irq_issue,
    error_library_not_found,
    error_function_not_found,
    error_corrupted_inforom,
    error_gpu_is_lost,
    error_reset_required,
    error_operating_system,
    error_lib_rm_version_mismatch,
    error_in_use,
    error_memory,
    error_no_data,
    error_vgpu_ecc_not_supported,
    error_insufficient_resources,
    error_freq_not_supported,
    error_argument_version_mismatch,
    error_deprecated,
    error_not_ready,
    error_gpu_not_found,
    error_invalid_state,
    error_reset_type_not_supported,
    error_unknown,
};

fn check(ret: c_uint) Error!void {
    if (ret == c.NVML_SUCCESS) return;
    return switch (ret) {
        1 => error.error_uninitialized,
        2 => error.error_invalid_argument,
        3 => error.error_not_supported,
        4 => error.error_no_permission,
        5 => error.error_already_initialized,
        6 => error.error_not_found,
        7 => error.error_insufficient_size,
        8 => error.error_insufficient_power,
        9 => error.error_driver_not_loaded,
        10 => error.error_timeout,
        11 => error.error_irq_issue,
        12 => error.error_library_not_found,
        13 => error.error_function_not_found,
        14 => error.error_corrupted_inforom,
        15 => error.error_gpu_is_lost,
        16 => error.error_reset_required,
        17 => error.error_operating_system,
        18 => error.error_lib_rm_version_mismatch,
        19 => error.error_in_use,
        20 => error.error_memory,
        21 => error.error_no_data,
        22 => error.error_vgpu_ecc_not_supported,
        23 => error.error_insufficient_resources,
        24 => error.error_freq_not_supported,
        25 => error.error_argument_version_mismatch,
        26 => error.error_deprecated,
        27 => error.error_not_ready,
        28 => error.error_gpu_not_found,
        29 => error.error_invalid_state,
        30 => error.error_reset_type_not_supported,
        999 => error.error_unknown,
        else => error.error_unknown,
    };
}

var lib: DynLib = .{};

fn call(comptime name: [:0]const u8, args: std.meta.ArgsTuple(@TypeOf(@field(c, name)))) Error!void {
    const f = lib.sym(name) orelse return error.NvmlUnavailable;
    try check(@call(.auto, f, args));
}

// Public API

pub fn init() Error!void {
    if (!lib.open("libnvidia-ml.so.1") and !lib.open("libnvidia-ml.so"))
        return error.NvmlUnavailable;
    try call("nvmlInit_v2", .{});
}

pub fn getHandleByIndex(device_id: u32) Error!c.nvmlDevice_t {
    var handle: c.nvmlDevice_t = undefined;
    try call("nvmlDeviceGetHandleByIndex_v2", .{ device_id, &handle });
    return handle;
}

pub fn getDeviceCount() Error!u32 {
    var count: c_uint = 0;
    try call("nvmlDeviceGetCount_v2", .{&count});
    return @intCast(count);
}

pub fn getName(handle: c.nvmlDevice_t, buf: []u8) Error![:0]const u8 {
    try call("nvmlDeviceGetName", .{ handle, buf.ptr, @as(c_uint, @intCast(buf.len)) });
    return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(buf.ptr)), 0);
}

pub fn getPowerUsage(handle: c.nvmlDevice_t) Error!c_uint {
    var power_mw: c_uint = 0;
    try call("nvmlDeviceGetPowerUsage", .{ handle, &power_mw });
    return power_mw;
}

pub fn getTemperature(handle: c.nvmlDevice_t) Error!c_uint {
    var temp: c_uint = 0;
    try call("nvmlDeviceGetTemperature", .{ handle, c.NVML_TEMPERATURE_GPU, &temp });
    return temp;
}

pub fn getUtilizationGpu(handle: c.nvmlDevice_t) Error!c_uint {
    var util: c.nvmlUtilization_t = undefined;
    try call("nvmlDeviceGetUtilizationRates", .{ handle, &util });
    return util.gpu;
}

pub fn getClockGraphics(handle: c.nvmlDevice_t) Error!c_uint {
    var clock: c_uint = 0;
    try call("nvmlDeviceGetClockInfo", .{ handle, c.NVML_CLOCK_GRAPHICS, &clock });
    return clock;
}

pub fn getClockSm(handle: c.nvmlDevice_t) Error!c_uint {
    var clock: c_uint = 0;
    try call("nvmlDeviceGetClockInfo", .{ handle, c.NVML_CLOCK_SM, &clock });
    return clock;
}

pub fn getClockMem(handle: c.nvmlDevice_t) Error!c_uint {
    var clock: c_uint = 0;
    try call("nvmlDeviceGetClockInfo", .{ handle, c.NVML_CLOCK_MEM, &clock });
    return clock;
}

pub fn getMaxClockGraphics(handle: c.nvmlDevice_t) Error!c_uint {
    var clock: c_uint = 0;
    try call("nvmlDeviceGetMaxClockInfo", .{ handle, c.NVML_CLOCK_GRAPHICS, &clock });
    return clock;
}

pub fn getMaxClockMem(handle: c.nvmlDevice_t) Error!c_uint {
    var clock: c_uint = 0;
    try call("nvmlDeviceGetMaxClockInfo", .{ handle, c.NVML_CLOCK_MEM, &clock });
    return clock;
}

pub fn getMemTotal(handle: c.nvmlDevice_t) Error!u64 {
    var mem: c.nvmlMemory_t = undefined;
    try call("nvmlDeviceGetMemoryInfo", .{ handle, &mem });
    return @intCast(mem.total);
}

pub fn getMemUsed(handle: c.nvmlDevice_t) Error!u64 {
    var mem: c.nvmlMemory_t = undefined;
    try call("nvmlDeviceGetMemoryInfo", .{ handle, &mem });
    return @intCast(mem.used);
}

pub fn getFanSpeed(handle: c.nvmlDevice_t) Error!c_uint {
    var speed: c_uint = 0;
    try call("nvmlDeviceGetFanSpeed", .{ handle, &speed });
    return speed;
}

pub fn getPowerLimit(handle: c.nvmlDevice_t) Error!c_uint {
    var limit: c_uint = 0;
    try call("nvmlDeviceGetEnforcedPowerLimit", .{ handle, &limit });
    return limit;
}

pub fn getPcieTxKBps(handle: c.nvmlDevice_t) Error!c_uint {
    var value: c_uint = 0;
    try call("nvmlDeviceGetPcieThroughput", .{ handle, c.NVML_PCIE_UTIL_TX_BYTES, &value });
    return value;
}

pub fn getPcieRxKBps(handle: c.nvmlDevice_t) Error!c_uint {
    var value: c_uint = 0;
    try call("nvmlDeviceGetPcieThroughput", .{ handle, c.NVML_PCIE_UTIL_RX_BYTES, &value });
    return value;
}

pub fn getEncoderUtil(handle: c.nvmlDevice_t) Error!c_uint {
    var util: c_uint = 0;
    var sampling: c_uint = 0;
    try call("nvmlDeviceGetEncoderUtilization", .{ handle, &util, &sampling });
    return util;
}

pub fn getDecoderUtil(handle: c.nvmlDevice_t) Error!c_uint {
    var util: c_uint = 0;
    var sampling: c_uint = 0;
    try call("nvmlDeviceGetDecoderUtilization", .{ handle, &util, &sampling });
    return util;
}

pub fn getTotalEnergy(handle: c.nvmlDevice_t) Error!u64 {
    var energy: c_ulonglong = 0;
    try call("nvmlDeviceGetTotalEnergyConsumption", .{ handle, &energy });
    return @intCast(energy);
}

pub fn getPcieLinkGen(handle: c.nvmlDevice_t) Error!c_uint {
    var gen: c_uint = 0;
    try call("nvmlDeviceGetCurrPcieLinkGeneration", .{ handle, &gen });
    return gen;
}

pub fn getPcieLinkWidth(handle: c.nvmlDevice_t) Error!c_uint {
    var width: c_uint = 0;
    try call("nvmlDeviceGetCurrPcieLinkWidth", .{ handle, &width });
    return width;
}

pub fn getMemBusWidth(handle: c.nvmlDevice_t) Error!c_uint {
    var width: c_uint = 0;
    try call("nvmlDeviceGetMemoryBusWidth", .{ handle, &width });
    return width;
}

pub fn getComputeRunningProcesses(allocator: std.mem.Allocator, handle: c.nvmlDevice_t) (Error || error{OutOfMemory})![]c.nvmlProcessInfo_t {
    const f = lib.sym("nvmlDeviceGetComputeRunningProcesses_v3") orelse return error.NvmlUnavailable;
    var count: c_uint = 0;
    const ret = f(handle, &count, null);
    if (ret != c.NVML_ERROR_INSUFFICIENT_SIZE) {
        if (ret == c.NVML_SUCCESS) return &.{};
        try check(ret);
    }
    if (count == 0) return &.{};
    const infos = try allocator.alloc(c.nvmlProcessInfo_t, count);
    errdefer allocator.free(infos);
    try check(f(handle, &count, @ptrCast(infos.ptr)));
    return infos;
}

pub fn getGraphicsRunningProcesses(allocator: std.mem.Allocator, handle: c.nvmlDevice_t) (Error || error{OutOfMemory})![]const c.nvmlProcessInfo_t {
    const f = lib.sym("nvmlDeviceGetGraphicsRunningProcesses_v3") orelse return error.NvmlUnavailable;

    var count: c_uint = 0;
    const ret = f(handle, &count, null);

    if (ret != c.NVML_ERROR_INSUFFICIENT_SIZE) {
        if (ret == c.NVML_SUCCESS) return &.{};
        try check(ret);
    }

    if (count == 0) return &.{};

    const infos = try allocator.alloc(c.nvmlProcessInfo_t, count);
    errdefer allocator.free(infos);

    try check(f(handle, &count, @ptrCast(infos.ptr)));

    return infos;
}

pub fn getProcessUtilization(allocator: std.mem.Allocator, handle: c.nvmlDevice_t, last_seen: u64) (Error || error{OutOfMemory})![]const c.nvmlProcessUtilizationSample_t {
    const f = lib.sym("nvmlDeviceGetProcessUtilization") orelse return error.NvmlUnavailable;

    var count: c_uint = 0;
    const ret = f(handle, null, &count, last_seen);

    if (ret != c.NVML_ERROR_INSUFFICIENT_SIZE) {
        if (ret == c.NVML_SUCCESS) return &.{};
        try check(ret);
    }

    if (count == 0) return &.{};

    const samples = try allocator.alloc(c.nvmlProcessUtilizationSample_t, count);
    errdefer allocator.free(samples);

    try check(f(handle, @ptrCast(samples.ptr), &count, last_seen));

    return samples;
}
