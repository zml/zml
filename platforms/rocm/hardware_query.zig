const std = @import("std");

const c = @import("c");

const Capabilities = @import("Capabilities.zig");

pub fn query(driver: *std.DynLib, ordinal: i32) !Capabilities.Hardware.Properties {
    if (ordinal < 0) return error.InvalidDevice;

    const hipInit = driver.lookup(@TypeOf(&c.hipInit), "hipInit") orelse return error.Unavailable;
    const hipDeviceGet = driver.lookup(@TypeOf(&c.hipDeviceGet), "hipDeviceGet") orelse return error.Unavailable;
    const hipDeviceGetAttribute = driver.lookup(@TypeOf(&c.hipDeviceGetAttribute), "hipDeviceGetAttribute") orelse return error.Unavailable;
    const hipDeviceTotalMem = driver.lookup(@TypeOf(&c.hipDeviceTotalMem), "hipDeviceTotalMem") orelse return error.Unavailable;

    if (hipInit(0) != c.hipSuccess) return error.Unavailable;
    var device: c.hipDevice_t = undefined;
    if (hipDeviceGet(&device, ordinal) != c.hipSuccess) return error.InvalidDevice;
    const attrs: DeviceQuery = .{ .device = device, .getAttribute = hipDeviceGetAttribute };
    var memoryBytes: usize = 0;
    const hasMemory = hipDeviceTotalMem(&memoryBytes, device) == c.hipSuccess;
    return .{
        .multiprocessorCount = attrs.uint(c.hipDeviceAttributeMultiprocessorCount),
        .maxThreadsPerBlockLimit = attrs.uint(c.hipDeviceAttributeMaxThreadsPerBlock),
        .sharedMemoryPerBlockLimit = attrs.uint(c.hipDeviceAttributeMaxSharedMemoryPerBlock),
        .l2CacheBytes = if (attrs.uint(c.hipDeviceAttributeL2CacheSize)) |bytes| @as(u64, bytes) else null,
        .globalMemoryBytes = if (hasMemory) memoryBytes else null,
        .memoryBusWidthBits = attrs.uint(c.hipDeviceAttributeMemoryBusWidth),
        .clockRateKHz = attrs.uint(c.hipDeviceAttributeClockRate),
        .memoryClockRateKHz = attrs.uint(c.hipDeviceAttributeMemoryClockRate),
        .asyncEngineCount = attrs.uint(c.hipDeviceAttributeAsyncEngineCount),
        .concurrentKernels = attrs.boolean(c.hipDeviceAttributeConcurrentKernels),
        .cooperativeLaunch = attrs.boolean(c.hipDeviceAttributeCooperativeLaunch),
        .integrated = attrs.boolean(c.hipDeviceAttributeIntegrated),
        .eccEnabled = attrs.boolean(c.hipDeviceAttributeEccEnabled),
        .xccCount = attrs.uint(c.hipDeviceAttributeNumberOfXccs),
    };
}

const DeviceQuery = struct {
    device: c.hipDevice_t,
    getAttribute: @TypeOf(&c.hipDeviceGetAttribute),

    fn uint(self: DeviceQuery, attribute: c.hipDeviceAttribute_t) ?u32 {
        var value: c_int = 0;
        // Unsupported attributes remain unknown; a successful zero is a fact.
        if (self.getAttribute(&value, attribute, self.device) != c.hipSuccess) return null;
        return std.math.cast(u32, value);
    }

    fn boolean(self: DeviceQuery, attribute: c.hipDeviceAttribute_t) ?bool {
        return (self.uint(attribute) orelse return null) != 0;
    }
};
