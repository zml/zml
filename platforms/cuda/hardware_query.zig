const std = @import("std");

const c = @import("c");

const Capabilities = @import("Capabilities.zig");

pub fn query(driver: *std.DynLib, ordinal: i32) !Capabilities.Hardware.Properties {
    if (ordinal < 0) return error.InvalidDevice;

    const cuInit = driver.lookup(@TypeOf(&c.cuInit), "cuInit") orelse return error.Unavailable;
    const cuDeviceGet = driver.lookup(@TypeOf(&c.cuDeviceGet), "cuDeviceGet") orelse return error.Unavailable;
    const cuDeviceGetAttribute = driver.lookup(@TypeOf(&c.cuDeviceGetAttribute), "cuDeviceGetAttribute") orelse return error.Unavailable;
    const cuDeviceTotalMem_v2 = driver.lookup(@TypeOf(&c.cuDeviceTotalMem_v2), "cuDeviceTotalMem_v2") orelse return error.Unavailable;

    if (cuInit(0) != c.CUDA_SUCCESS) return error.Unavailable;
    var device: c.CUdevice = undefined;
    if (cuDeviceGet(&device, ordinal) != c.CUDA_SUCCESS) return error.InvalidDevice;
    const attrs: DeviceQuery = .{ .device = device, .getAttribute = cuDeviceGetAttribute };
    var memoryBytes: usize = 0;
    const hasMemory = cuDeviceTotalMem_v2(&memoryBytes, device) == c.CUDA_SUCCESS;
    return .{
        .multiprocessorCount = attrs.uint(c.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        .sharedMemoryPerBlockOptinLimit = attrs.uint(c.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),
        .reservedSharedMemoryPerBlock = attrs.uint(c.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK),
        .l2CacheBytes = attrs.uint(c.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
        .persistingL2CacheBytes = attrs.uint(c.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE),
        .accessPolicyWindowBytes = attrs.uint(c.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE),
        .globalMemoryBytes = if (hasMemory) memoryBytes else null,
        .memoryBusWidthBits = attrs.uint(c.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH),
        .typicalClockRateKHz = attrs.uint(c.CU_DEVICE_ATTRIBUTE_CLOCK_RATE),
        .peakMemoryClockRateKHz = attrs.uint(c.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE),
        .asyncEngineCount = attrs.uint(c.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT),
        .concurrentKernels = attrs.boolean(c.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS),
        .cooperativeLaunch = attrs.boolean(c.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH),
        .clusterLaunch = attrs.boolean(c.CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH),
        .integrated = attrs.boolean(c.CU_DEVICE_ATTRIBUTE_INTEGRATED),
        .eccEnabled = attrs.boolean(c.CU_DEVICE_ATTRIBUTE_ECC_ENABLED),
    };
}

const DeviceQuery = struct {
    device: c.CUdevice,
    getAttribute: @TypeOf(&c.cuDeviceGetAttribute),

    fn uint(self: DeviceQuery, attribute: c.CUdevice_attribute) ?u32 {
        var value: c_int = 0;
        // Older drivers may not know newer attributes. Preserve that distinction
        // from a successful query returning zero (e.g. no persisting L2 in MIG).
        if (self.getAttribute(&value, attribute, self.device) != c.CUDA_SUCCESS) return null;
        return std.math.cast(u32, value);
    }

    fn boolean(self: DeviceQuery, attribute: c.CUdevice_attribute) ?bool {
        return (self.uint(attribute) orelse return null) != 0;
    }
};
