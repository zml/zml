const std = @import("std");
const c = @import("c");
const DynLib = @import("../dynlib.zig");

const Nvml = @This();

pub const Error = ReturnError || error{NvmlUnavailable};
pub const Handle = c.nvmlDevice_t;
pub const ProcessInfo_t = c.nvmlProcessInfo_t;

lib: Fns,

const Fns = struct {
    nvmlInit_v2: DynLib.Fn("nvmlInit_v2"),
    nvmlDeviceGetHandleByIndex_v2: DynLib.Fn("nvmlDeviceGetHandleByIndex_v2"),
    nvmlDeviceGetCount_v2: DynLib.Fn("nvmlDeviceGetCount_v2"),
    nvmlDeviceGetName: DynLib.Fn("nvmlDeviceGetName"),
    nvmlDeviceGetPowerUsage: DynLib.Fn("nvmlDeviceGetPowerUsage"),
    nvmlDeviceGetTemperature: DynLib.Fn("nvmlDeviceGetTemperature"),
    nvmlDeviceGetUtilizationRates: DynLib.Fn("nvmlDeviceGetUtilizationRates"),
    nvmlDeviceGetClockInfo: DynLib.Fn("nvmlDeviceGetClockInfo"),
    nvmlDeviceGetMaxClockInfo: DynLib.Fn("nvmlDeviceGetMaxClockInfo"),
    nvmlDeviceGetMemoryInfo: DynLib.Fn("nvmlDeviceGetMemoryInfo"),
    nvmlDeviceGetFanSpeed: DynLib.Fn("nvmlDeviceGetFanSpeed"),
    nvmlDeviceGetEnforcedPowerLimit: DynLib.Fn("nvmlDeviceGetEnforcedPowerLimit"),
    nvmlDeviceGetPcieThroughput: DynLib.Fn("nvmlDeviceGetPcieThroughput"),
    nvmlDeviceGetEncoderUtilization: DynLib.Fn("nvmlDeviceGetEncoderUtilization"),
    nvmlDeviceGetDecoderUtilization: DynLib.Fn("nvmlDeviceGetDecoderUtilization"),
    nvmlDeviceGetTotalEnergyConsumption: DynLib.Fn("nvmlDeviceGetTotalEnergyConsumption"),
    nvmlDeviceGetCurrPcieLinkGeneration: DynLib.Fn("nvmlDeviceGetCurrPcieLinkGeneration"),
    nvmlDeviceGetCurrPcieLinkWidth: DynLib.Fn("nvmlDeviceGetCurrPcieLinkWidth"),
    nvmlDeviceGetMemoryBusWidth: DynLib.Fn("nvmlDeviceGetMemoryBusWidth"),
    nvmlDeviceGetComputeRunningProcesses_v3: DynLib.Fn("nvmlDeviceGetComputeRunningProcesses_v3"),
    nvmlDeviceGetGraphicsRunningProcesses_v3: DynLib.Fn("nvmlDeviceGetGraphicsRunningProcesses_v3"),
    nvmlDeviceGetProcessUtilization: DynLib.Fn("nvmlDeviceGetProcessUtilization"),
};

pub fn init() Error!Nvml {
    const fns = DynLib.open(Fns, "libnvidia-ml.so.1") orelse
        DynLib.open(Fns, "libnvidia-ml.so") orelse
        return error.NvmlUnavailable;
    try check(fns.nvmlInit_v2());
    return .{ .lib = fns };
}

pub fn handleByIndex(self: Nvml, device_id: u32) Error!c.nvmlDevice_t {
    var handle: c.nvmlDevice_t = undefined;
    try check(self.lib.nvmlDeviceGetHandleByIndex_v2(device_id, &handle));
    return handle;
}

pub fn deviceCount(self: Nvml) Error!u32 {
    var count: c_uint = 0;
    try check(self.lib.nvmlDeviceGetCount_v2(&count));
    return @intCast(count);
}

pub fn name(self: Nvml, handle: c.nvmlDevice_t, buf: *[256]u8) Error![:0]const u8 {
    try check(self.lib.nvmlDeviceGetName(handle, buf, buf.len));
    return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(buf)), 0);
}

pub fn powerUsage(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var power_mw: c_uint = 0;
    try check(self.lib.nvmlDeviceGetPowerUsage(handle, &power_mw));
    return power_mw;
}

pub fn temperature(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var temp: c_uint = 0;
    try check(self.lib.nvmlDeviceGetTemperature(handle, c.NVML_TEMPERATURE_GPU, &temp));
    return temp;
}

pub fn utilizationGpu(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var util: c.nvmlUtilization_t = undefined;
    try check(self.lib.nvmlDeviceGetUtilizationRates(handle, &util));
    return util.gpu;
}

pub fn clockGraphics(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var clock: c_uint = 0;
    try check(self.lib.nvmlDeviceGetClockInfo(handle, c.NVML_CLOCK_GRAPHICS, &clock));
    return clock;
}

pub fn clockSm(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var clock: c_uint = 0;
    try check(self.lib.nvmlDeviceGetClockInfo(handle, c.NVML_CLOCK_SM, &clock));
    return clock;
}

pub fn clockMem(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var clock: c_uint = 0;
    try check(self.lib.nvmlDeviceGetClockInfo(handle, c.NVML_CLOCK_MEM, &clock));
    return clock;
}

pub fn maxClockGraphics(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var clock: c_uint = 0;
    try check(self.lib.nvmlDeviceGetMaxClockInfo(handle, c.NVML_CLOCK_GRAPHICS, &clock));
    return clock;
}

pub fn maxClockMem(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var clock: c_uint = 0;
    try check(self.lib.nvmlDeviceGetMaxClockInfo(handle, c.NVML_CLOCK_MEM, &clock));
    return clock;
}

pub fn memTotal(self: Nvml, handle: c.nvmlDevice_t) Error!u64 {
    var mem: c.nvmlMemory_t = undefined;
    try check(self.lib.nvmlDeviceGetMemoryInfo(handle, &mem));
    return @intCast(mem.total);
}

pub fn memUsed(self: Nvml, handle: c.nvmlDevice_t) Error!u64 {
    var mem: c.nvmlMemory_t = undefined;
    try check(self.lib.nvmlDeviceGetMemoryInfo(handle, &mem));
    return @intCast(mem.used);
}

pub fn fanSpeed(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var speed: c_uint = 0;
    try check(self.lib.nvmlDeviceGetFanSpeed(handle, &speed));
    return speed;
}

pub fn powerLimit(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var limit: c_uint = 0;
    try check(self.lib.nvmlDeviceGetEnforcedPowerLimit(handle, &limit));
    return limit;
}

pub fn pcieTxKBps(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var value: c_uint = 0;
    try check(self.lib.nvmlDeviceGetPcieThroughput(handle, c.NVML_PCIE_UTIL_TX_BYTES, &value));
    return value;
}

pub fn pcieRxKBps(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var value: c_uint = 0;
    try check(self.lib.nvmlDeviceGetPcieThroughput(handle, c.NVML_PCIE_UTIL_RX_BYTES, &value));
    return value;
}

pub fn encoderUtil(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var util: c_uint = 0;
    var sampling: c_uint = 0;
    try check(self.lib.nvmlDeviceGetEncoderUtilization(handle, &util, &sampling));
    return util;
}

pub fn decoderUtil(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var util: c_uint = 0;
    var sampling: c_uint = 0;
    try check(self.lib.nvmlDeviceGetDecoderUtilization(handle, &util, &sampling));
    return util;
}

pub fn totalEnergy(self: Nvml, handle: c.nvmlDevice_t) Error!u64 {
    var energy: c_ulonglong = 0;
    try check(self.lib.nvmlDeviceGetTotalEnergyConsumption(handle, &energy));
    return @intCast(energy);
}

pub fn pcieLinkGen(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var gen: c_uint = 0;
    try check(self.lib.nvmlDeviceGetCurrPcieLinkGeneration(handle, &gen));
    return gen;
}

pub fn pcieLinkWidth(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var width: c_uint = 0;
    try check(self.lib.nvmlDeviceGetCurrPcieLinkWidth(handle, &width));
    return width;
}

pub fn memBusWidth(self: Nvml, handle: c.nvmlDevice_t) Error!c_uint {
    var width: c_uint = 0;
    try check(self.lib.nvmlDeviceGetMemoryBusWidth(handle, &width));
    return width;
}

pub fn computeRunningProcesses(self: Nvml, allocator: std.mem.Allocator, handle: c.nvmlDevice_t) (Error || error{OutOfMemory})![]c.nvmlProcessInfo_t {
    var count: c_uint = 0;
    const ret = self.lib.nvmlDeviceGetComputeRunningProcesses_v3(handle, &count, null);
    if (ret != c.NVML_ERROR_INSUFFICIENT_SIZE) {
        if (ret == c.NVML_SUCCESS) {
            return &.{};
        }
        try check(ret);
    }
    if (count == 0) {
        return &.{};
    }
    const infos = try allocator.alloc(c.nvmlProcessInfo_t, count);
    errdefer allocator.free(infos);
    try check(self.lib.nvmlDeviceGetComputeRunningProcesses_v3(handle, &count, @ptrCast(infos.ptr)));
    return infos;
}

pub fn graphicsRunningProcesses(self: Nvml, allocator: std.mem.Allocator, handle: c.nvmlDevice_t) (Error || error{OutOfMemory})![]const c.nvmlProcessInfo_t {
    var count: c_uint = 0;
    const ret = self.lib.nvmlDeviceGetGraphicsRunningProcesses_v3(handle, &count, null);

    if (ret != c.NVML_ERROR_INSUFFICIENT_SIZE) {
        if (ret == c.NVML_SUCCESS) {
            return &.{};
        }
        try check(ret);
    }

    if (count == 0) {
        return &.{};
    }

    const infos = try allocator.alloc(c.nvmlProcessInfo_t, count);
    errdefer allocator.free(infos);

    try check(self.lib.nvmlDeviceGetGraphicsRunningProcesses_v3(handle, &count, @ptrCast(infos.ptr)));

    return infos;
}

pub fn processUtilization(self: Nvml, allocator: std.mem.Allocator, handle: c.nvmlDevice_t, last_seen: u64) (Error || error{OutOfMemory})![]const c.nvmlProcessUtilizationSample_t {
    var count: c_uint = 0;
    const ret = self.lib.nvmlDeviceGetProcessUtilization(handle, null, &count, last_seen);

    if (ret != c.NVML_ERROR_INSUFFICIENT_SIZE) {
        if (ret == c.NVML_SUCCESS) {
            return &.{};
        }
        try check(ret);
    }

    if (count == 0) {
        return &.{};
    }

    const samples = try allocator.alloc(c.nvmlProcessUtilizationSample_t, count);
    errdefer allocator.free(samples);

    try check(self.lib.nvmlDeviceGetProcessUtilization(handle, @ptrCast(samples.ptr), &count, last_seen));

    return samples;
}

// Error handling

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
    if (ret == c.NVML_SUCCESS) {
        return;
    }
    return switch (ret) {
        c.NVML_ERROR_UNINITIALIZED => error.error_uninitialized,
        c.NVML_ERROR_INVALID_ARGUMENT => error.error_invalid_argument,
        c.NVML_ERROR_NOT_SUPPORTED => error.error_not_supported,
        c.NVML_ERROR_NO_PERMISSION => error.error_no_permission,
        c.NVML_ERROR_ALREADY_INITIALIZED => error.error_already_initialized,
        c.NVML_ERROR_NOT_FOUND => error.error_not_found,
        c.NVML_ERROR_INSUFFICIENT_SIZE => error.error_insufficient_size,
        c.NVML_ERROR_INSUFFICIENT_POWER => error.error_insufficient_power,
        c.NVML_ERROR_DRIVER_NOT_LOADED => error.error_driver_not_loaded,
        c.NVML_ERROR_TIMEOUT => error.error_timeout,
        c.NVML_ERROR_IRQ_ISSUE => error.error_irq_issue,
        c.NVML_ERROR_LIBRARY_NOT_FOUND => error.error_library_not_found,
        c.NVML_ERROR_FUNCTION_NOT_FOUND => error.error_function_not_found,
        c.NVML_ERROR_CORRUPTED_INFOROM => error.error_corrupted_inforom,
        c.NVML_ERROR_GPU_IS_LOST => error.error_gpu_is_lost,
        c.NVML_ERROR_RESET_REQUIRED => error.error_reset_required,
        c.NVML_ERROR_OPERATING_SYSTEM => error.error_operating_system,
        c.NVML_ERROR_LIB_RM_VERSION_MISMATCH => error.error_lib_rm_version_mismatch,
        c.NVML_ERROR_IN_USE => error.error_in_use,
        c.NVML_ERROR_MEMORY => error.error_memory,
        c.NVML_ERROR_NO_DATA => error.error_no_data,
        c.NVML_ERROR_VGPU_ECC_NOT_SUPPORTED => error.error_vgpu_ecc_not_supported,
        c.NVML_ERROR_INSUFFICIENT_RESOURCES => error.error_insufficient_resources,
        c.NVML_ERROR_FREQ_NOT_SUPPORTED => error.error_freq_not_supported,
        c.NVML_ERROR_ARGUMENT_VERSION_MISMATCH => error.error_argument_version_mismatch,
        c.NVML_ERROR_DEPRECATED => error.error_deprecated,
        c.NVML_ERROR_NOT_READY => error.error_not_ready,
        c.NVML_ERROR_GPU_NOT_FOUND => error.error_gpu_not_found,
        c.NVML_ERROR_INVALID_STATE => error.error_invalid_state,
        c.NVML_ERROR_RESET_TYPE_NOT_SUPPORTED => error.error_reset_type_not_supported,
        c.NVML_ERROR_UNKNOWN => error.error_unknown,
        else => error.error_unknown,
    };
}
