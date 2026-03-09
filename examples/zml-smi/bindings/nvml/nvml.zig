const std = @import("std");
const c = @import("c");

const has_nvml = @hasDecl(c, "nvmlInit_v2");

pub const Handle = if (has_nvml) c.nvmlDevice_t else ?*opaque {};
pub const ProcessInfo_t = if (has_nvml) c.nvmlProcessInfo_t else ?*opaque {};
pub const ProcessUtilSample_t = if (has_nvml) c.nvmlProcessUtilizationSample_t else ?*opaque {};

pub const Error = ReturnError || error{NvmlUnavailable};

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

const NVML_SUCCESS: c_uint = if (has_nvml) c.NVML_SUCCESS else 0;
const NVML_INSUFFICIENT_SIZE: c_uint = if (has_nvml) c.NVML_ERROR_INSUFFICIENT_SIZE else 7;

fn check(ret: c_uint) Error!void {
    if (ret == NVML_SUCCESS) return;
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

// Public API

pub fn init() Error!void {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    try check(c.nvmlInit_v2());
}

pub fn getHandleByIndex(device_id: u32) Error!Handle {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var handle: Handle = undefined;
    try check(c.nvmlDeviceGetHandleByIndex_v2(device_id, &handle));
    return handle;
}

pub fn getDeviceCount() Error!u32 {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var count: c_uint = 0;
    try check(c.nvmlDeviceGetCount_v2(&count));
    return @intCast(count);
}

pub fn getName(handle: Handle, buf: []u8) Error![:0]const u8 {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    try check(c.nvmlDeviceGetName(handle, buf.ptr, @intCast(buf.len)));
    return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(buf.ptr)), 0);
}

pub fn getPowerUsage(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var power_mw: c_uint = 0;
    try check(c.nvmlDeviceGetPowerUsage(handle, &power_mw));
    return power_mw;
}

pub fn getTemperature(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var temp: c_uint = 0;
    try check(c.nvmlDeviceGetTemperature(handle, c.NVML_TEMPERATURE_GPU, &temp));
    return temp;
}

pub fn getUtilizationGpu(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var util: c.nvmlUtilization_t = undefined;
    try check(c.nvmlDeviceGetUtilizationRates(handle, &util));
    return util.gpu;
}

pub fn getClockGraphics(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var clock: c_uint = 0;
    try check(c.nvmlDeviceGetClockInfo(handle, c.NVML_CLOCK_GRAPHICS, &clock));
    return clock;
}

pub fn getClockSm(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var clock: c_uint = 0;
    try check(c.nvmlDeviceGetClockInfo(handle, c.NVML_CLOCK_SM, &clock));
    return clock;
}

pub fn getClockMem(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var clock: c_uint = 0;
    try check(c.nvmlDeviceGetClockInfo(handle, c.NVML_CLOCK_MEM, &clock));
    return clock;
}

pub fn getMaxClockGraphics(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var clock: c_uint = 0;
    try check(c.nvmlDeviceGetMaxClockInfo(handle, c.NVML_CLOCK_GRAPHICS, &clock));
    return clock;
}

pub fn getMaxClockMem(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var clock: c_uint = 0;
    try check(c.nvmlDeviceGetMaxClockInfo(handle, c.NVML_CLOCK_MEM, &clock));
    return clock;
}

pub fn getMemTotal(handle: Handle) Error!u64 {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var mem: c.nvmlMemory_t = undefined;
    try check(c.nvmlDeviceGetMemoryInfo(handle, &mem));
    return @intCast(mem.total);
}

pub fn getMemUsed(handle: Handle) Error!u64 {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var mem: c.nvmlMemory_t = undefined;
    try check(c.nvmlDeviceGetMemoryInfo(handle, &mem));
    return @intCast(mem.used);
}

pub fn getFanSpeed(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var speed: c_uint = 0;
    try check(c.nvmlDeviceGetFanSpeed(handle, &speed));
    return speed;
}

pub fn getPowerLimit(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var limit: c_uint = 0;
    try check(c.nvmlDeviceGetEnforcedPowerLimit(handle, &limit));
    return limit;
}

pub fn getPcieTxKBps(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var value: c_uint = 0;
    try check(c.nvmlDeviceGetPcieThroughput(handle, c.NVML_PCIE_UTIL_TX_BYTES, &value));
    return value;
}

pub fn getPcieRxKBps(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var value: c_uint = 0;
    try check(c.nvmlDeviceGetPcieThroughput(handle, c.NVML_PCIE_UTIL_RX_BYTES, &value));
    return value;
}

pub fn getEncoderUtil(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var util: c_uint = 0;
    var sampling: c_uint = 0;
    try check(c.nvmlDeviceGetEncoderUtilization(handle, &util, &sampling));
    return util;
}

pub fn getDecoderUtil(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var util: c_uint = 0;
    var sampling: c_uint = 0;
    try check(c.nvmlDeviceGetDecoderUtilization(handle, &util, &sampling));
    return util;
}

pub fn getTotalEnergy(handle: Handle) Error!u64 {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var energy: c_ulonglong = 0;
    try check(c.nvmlDeviceGetTotalEnergyConsumption(handle, &energy));
    return @intCast(energy);
}

pub fn getPcieLinkGen(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var gen: c_uint = 0;
    try check(c.nvmlDeviceGetCurrPcieLinkGeneration(handle, &gen));
    return gen;
}

pub fn getPcieLinkWidth(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var width: c_uint = 0;
    try check(c.nvmlDeviceGetCurrPcieLinkWidth(handle, &width));
    return width;
}

pub fn getMemBusWidth(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var width: c_uint = 0;
    try check(c.nvmlDeviceGetMemoryBusWidth(handle, &width));
    return width;
}

pub fn getNumFans(handle: Handle) Error!c_uint {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var count: c_uint = 0;
    try check(c.nvmlDeviceGetNumFans(handle, &count));
    return count;
}

pub fn getComputeRunningProcesses(handle: Handle, infos: []ProcessInfo_t) Error![]const ProcessInfo_t {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var count: c_uint = @intCast(infos.len);
    const ret = c.nvmlDeviceGetComputeRunningProcesses_v3(handle, &count, @ptrCast(infos.ptr));
    if (ret == NVML_INSUFFICIENT_SIZE) return infos[0..0];
    try check(ret);
    return infos[0..count];
}

pub fn getGraphicsRunningProcesses(handle: Handle, infos: []ProcessInfo_t) Error![]const ProcessInfo_t {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var count: c_uint = @intCast(infos.len);
    const ret = c.nvmlDeviceGetGraphicsRunningProcesses_v3(handle, &count, @ptrCast(infos.ptr));
    if (ret == NVML_INSUFFICIENT_SIZE) return infos[0..0];
    try check(ret);
    return infos[0..count];
}

pub fn getProcessUtilization(handle: Handle, samples: []ProcessUtilSample_t, last_seen: u64) Error![]const ProcessUtilSample_t {
    if (comptime !has_nvml) return error.NvmlUnavailable;
    var count: c_uint = 0;
    const ret = c.nvmlDeviceGetProcessUtilization(handle, null, &count, last_seen);
    if (ret != NVML_INSUFFICIENT_SIZE) {
        if (ret == NVML_SUCCESS) return samples[0..0];
        try check(ret);
    }
    if (count == 0) return samples[0..0];
    count = @min(count, @as(c_uint, @intCast(samples.len)));
    try check(c.nvmlDeviceGetProcessUtilization(handle, @ptrCast(samples.ptr), &count, last_seen));
    return samples[0..count];
}
