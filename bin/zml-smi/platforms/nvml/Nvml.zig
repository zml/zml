const std = @import("std");
const c = @import("c");
const smi = @import("zml-smi");

const Nvml = @This();
const log = std.log.scoped(.@"zml-smi/nvml");

platform: smi.Platform,
api: Api,
gpus: []Gpu,
tick: u8 = 0,

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes };

const Api = struct {
    nvmlInit_v2: *const @TypeOf(c.nvmlInit_v2),
    nvmlSystemGetDriverVersion: *const @TypeOf(c.nvmlSystemGetDriverVersion),
    nvmlSystemGetCudaDriverVersion_v2: *const @TypeOf(c.nvmlSystemGetCudaDriverVersion_v2),
    nvmlDeviceGetCount_v2: *const @TypeOf(c.nvmlDeviceGetCount_v2),
    nvmlDeviceGetHandleByIndex_v2: *const @TypeOf(c.nvmlDeviceGetHandleByIndex_v2),
    nvmlDeviceGetName: *const @TypeOf(c.nvmlDeviceGetName),
    nvmlDeviceGetPowerUsage: *const @TypeOf(c.nvmlDeviceGetPowerUsage),
    nvmlDeviceGetEnforcedPowerLimit: *const @TypeOf(c.nvmlDeviceGetEnforcedPowerLimit),
    nvmlDeviceGetTemperature: *const @TypeOf(c.nvmlDeviceGetTemperature),
    nvmlDeviceGetFanSpeed: *const @TypeOf(c.nvmlDeviceGetFanSpeed),
    nvmlDeviceGetUtilizationRates: *const @TypeOf(c.nvmlDeviceGetUtilizationRates),
    nvmlDeviceGetEncoderUtilization: *const @TypeOf(c.nvmlDeviceGetEncoderUtilization),
    nvmlDeviceGetDecoderUtilization: *const @TypeOf(c.nvmlDeviceGetDecoderUtilization),
    nvmlDeviceGetClockInfo: *const @TypeOf(c.nvmlDeviceGetClockInfo),
    nvmlDeviceGetMaxClockInfo: *const @TypeOf(c.nvmlDeviceGetMaxClockInfo),
    nvmlDeviceGetMemoryInfo: *const @TypeOf(c.nvmlDeviceGetMemoryInfo),
    nvmlDeviceGetMemoryBusWidth: *const @TypeOf(c.nvmlDeviceGetMemoryBusWidth),
    nvmlDeviceGetPcieThroughput: *const @TypeOf(c.nvmlDeviceGetPcieThroughput),
    nvmlDeviceGetPcieSpeed: *const @TypeOf(c.nvmlDeviceGetPcieSpeed),
    nvmlDeviceGetCurrPcieLinkGeneration: *const @TypeOf(c.nvmlDeviceGetCurrPcieLinkGeneration),
    nvmlDeviceGetCurrPcieLinkWidth: *const @TypeOf(c.nvmlDeviceGetCurrPcieLinkWidth),
    nvmlDeviceGetComputeRunningProcesses_v3: *const @TypeOf(c.nvmlDeviceGetComputeRunningProcesses_v3),
    nvmlDeviceGetGraphicsRunningProcesses_v3: *const @TypeOf(c.nvmlDeviceGetGraphicsRunningProcesses_v3),
    nvmlDeviceGetProcessUtilization: *const @TypeOf(c.nvmlDeviceGetProcessUtilization),
};

pub fn detect(io: std.Io) bool {
    return smi.sysfs.exists(io, "/dev/nvidiactl");
}

pub fn init(arena: std.mem.Allocator, _: std.mem.Allocator, _: std.Io) !Nvml {
    const library = std.c.dlopen("libnvidia-ml.so.1", .{ .LAZY = true, .GLOBAL = true, .NODELETE = true }) orelse
        std.c.dlopen("libnvidia-ml.so", .{ .LAZY = true, .GLOBAL = true, .NODELETE = true }) orelse {
        if (std.c.dlerror()) |err| log.warn("dlopen: {s}", .{err});
        return error.LibraryNotFound;
    };

    const api = try smi.dynlib.lookupStruct(library, Api);
    try check(api.nvmlInit_v2());

    var version: [c.NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE]u8 = undefined;
    const driver_version = if (ok(api.nvmlSystemGetDriverVersion(&version, version.len)))
        try arena.dupe(u8, std.mem.sliceTo(&version, 0))
    else
        null;

    var cuda: c_int = 0;
    const details: []const smi.Device.Detail = if (ok(api.nvmlSystemGetCudaDriverVersion_v2(&cuda))) try arena.dupe(smi.Device.Detail, &.{.{
        .key = "cuda_version",
        .label = "CUDA",
        .value = try std.fmt.allocPrint(arena, "{d}.{d}", .{ @divTrunc(cuda, 1000), @divTrunc(@mod(cuda, 1000), 10) }),
        .in_title = true,
    }}) else &.{};

    var count: c_uint = 0;
    try check(api.nvmlDeviceGetCount_v2(&count));
    var gpus: std.ArrayList(Gpu) = try .initCapacity(arena, count);
    var devices: std.ArrayList(smi.Device) = try .initCapacity(arena, count);
    for (0..count) |index| {
        var handle: c.nvmlDevice_t = undefined;
        if (!ok(api.nvmlDeviceGetHandleByIndex_v2(@intCast(index), &handle))) continue;
        var name: [c.NVML_DEVICE_NAME_V2_BUFFER_SIZE]u8 = undefined;
        gpus.appendAssumeCapacity(.{ .handle = handle });
        devices.appendAssumeCapacity(.{
            .target = .cuda,
            .name = if (ok(api.nvmlDeviceGetName(handle, &name, name.len))) try arena.dupe(u8, std.mem.sliceTo(&name, 0)) else null,
            .driver_version = driver_version,
            .details = details,
        });
    }

    return .{
        .platform = .{ .devices = devices.items, .vtable = &vtable },
        .api = api,
        .gpus = gpus.items,
    };
}

fn poll(platform: *smi.Platform, _: std.mem.Allocator, samples: []smi.Sample) void {
    const self: *Nvml = @alignCast(@fieldParentPtr("platform", platform));
    const api = &self.api;

    // PCIe throughput blocks while it measures
    const read_pcie = self.tick % 4 == 0;
    self.tick +%= 1;

    for (self.gpus, samples) |*gpu, *sample| {
        const fields = &sample.cuda;
        const handle = gpu.handle;

        var value: c_uint = 0;
        fields.power_mw = if (ok(api.nvmlDeviceGetPowerUsage(handle, &value))) value else null;
        fields.power_limit_mw = if (ok(api.nvmlDeviceGetEnforcedPowerLimit(handle, &value))) value else null;
        fields.temperature = if (ok(api.nvmlDeviceGetTemperature(handle, c.NVML_TEMPERATURE_GPU, &value))) value else null;
        fields.fan_speed_percent = if (ok(api.nvmlDeviceGetFanSpeed(handle, &value))) value else null;
        var rates: c.nvmlUtilization_t = undefined;
        fields.util_percent = if (ok(api.nvmlDeviceGetUtilizationRates(handle, &rates))) rates.gpu else null;

        var util: c_uint, var period: c_uint = .{ 0, 0 };
        fields.encoder_util_percent = if (ok(api.nvmlDeviceGetEncoderUtilization(handle, &util, &period))) util else null;
        fields.decoder_util_percent = if (ok(api.nvmlDeviceGetDecoderUtilization(handle, &util, &period))) util else null;

        fields.clock_graphics_mhz = if (ok(api.nvmlDeviceGetClockInfo(handle, c.NVML_CLOCK_GRAPHICS, &value))) value else null;
        fields.clock_sm_mhz = if (ok(api.nvmlDeviceGetClockInfo(handle, c.NVML_CLOCK_SM, &value))) value else null;
        fields.clock_mem_mhz = if (ok(api.nvmlDeviceGetClockInfo(handle, c.NVML_CLOCK_MEM, &value))) value else null;
        fields.clock_graphics_max_mhz = if (ok(api.nvmlDeviceGetMaxClockInfo(handle, c.NVML_CLOCK_GRAPHICS, &value))) value else null;
        fields.clock_mem_max_mhz = if (ok(api.nvmlDeviceGetMaxClockInfo(handle, c.NVML_CLOCK_MEM, &value))) value else null;

        var memory: c.nvmlMemory_t = undefined;
        if (ok(api.nvmlDeviceGetMemoryInfo(handle, &memory))) {
            fields.mem_used_bytes = memory.used;
            fields.mem_total_bytes = memory.total;
        }
        fields.mem_bus_width = if (ok(api.nvmlDeviceGetMemoryBusWidth(handle, &value))) value else null;

        fields.pcie_link_gen = if (ok(api.nvmlDeviceGetCurrPcieLinkGeneration(handle, &value))) value else null;
        fields.pcie_link_width = if (ok(api.nvmlDeviceGetCurrPcieLinkWidth(handle, &value))) value else null;
        // NVML gives the speed of one lane.
        if (ok(api.nvmlDeviceGetPcieSpeed(handle, &value))) {
            if (fields.pcie_link_width) |width| fields.pcie_bandwidth_mbps = value * width;
        }

        if (read_pcie) {
            gpu.pcie_tx_kbps = if (ok(api.nvmlDeviceGetPcieThroughput(handle, c.NVML_PCIE_UTIL_TX_BYTES, &value))) value else null;
            gpu.pcie_rx_kbps = if (ok(api.nvmlDeviceGetPcieThroughput(handle, c.NVML_PCIE_UTIL_RX_BYTES, &value))) value else null;
        }
        fields.pcie_tx_kbps = gpu.pcie_tx_kbps;
        fields.pcie_rx_kbps = gpu.pcie_rx_kbps;
    }
}

fn processes(platform: *smi.Platform, scratch: std.mem.Allocator, _: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
    const self: *Nvml = @alignCast(@fieldParentPtr("platform", platform));
    var read_any = self.gpus.len == 0;

    for (self.gpus, 0..) |*gpu, device| {
        const handle = gpu.handle;
        const first_row = out.rows.items.len;
        inline for (.{ self.api.nvmlDeviceGetComputeRunningProcesses_v3, self.api.nvmlDeviceGetGraphicsRunningProcesses_v3 }) |query| {
            if (running(scratch, query, handle)) |list| {
                read_any = true;
                for (list) |process| {
                    try out.append(.{ .pid = process.pid, .device_idx = @intCast(device), .dev_mem_kib = process.usedGpuMemory / 1024 });
                }
            } else |_| {}
        }

        const since = gpu.last_seen;
        const utilization = self.processUtilization(scratch, handle, since) catch continue;
        for (utilization) |util| {
            if (util.smUtil > 100 or util.timeStamp <= since) continue;
            gpu.last_seen = @max(gpu.last_seen, util.timeStamp);
            for (out.rows.items[first_row..]) |*row| {
                if (row.pid == util.pid) row.dev_util_percent = @intCast(util.smUtil);
            }
        }
    }

    if (!read_any) return error.Unavailable;
}

const Gpu = struct {
    handle: c.nvmlDevice_t,
    pcie_tx_kbps: ?u64 = null,
    pcie_rx_kbps: ?u64 = null,
    last_seen: u64 = 0,
};

fn running(scratch: std.mem.Allocator, query: anytype, handle: c.nvmlDevice_t) ![]const c.nvmlProcessInfo_t {
    var count: c_uint = 0;
    const status = query(handle, &count, null);
    if (ok(status)) return &.{};
    if (status != c.NVML_ERROR_INSUFFICIENT_SIZE) try check(status);

    const list = try scratch.alloc(c.nvmlProcessInfo_t, count);
    try check(query(handle, &count, list.ptr));
    return list[0..count];
}

fn processUtilization(self: *Nvml, scratch: std.mem.Allocator, handle: c.nvmlDevice_t, since: u64) ![]const c.nvmlProcessUtilizationSample_t {
    var count: c_uint = 0;
    const status = self.api.nvmlDeviceGetProcessUtilization(handle, null, &count, since);
    if (ok(status)) return &.{};
    if (status != c.NVML_ERROR_INSUFFICIENT_SIZE) try check(status);

    const samples = try scratch.alloc(c.nvmlProcessUtilizationSample_t, count);
    try check(self.api.nvmlDeviceGetProcessUtilization(handle, samples.ptr, &count, since));
    return samples[0..count];
}

fn ok(status: c.nvmlReturn_t) bool {
    return status == c.NVML_SUCCESS;
}

fn check(status: c.nvmlReturn_t) !void {
    return switch (status) {
        c.NVML_SUCCESS => {},
        c.NVML_ERROR_UNINITIALIZED => error.Uninitialized,
        c.NVML_ERROR_INVALID_ARGUMENT => error.InvalidArgument,
        c.NVML_ERROR_NOT_SUPPORTED => error.NotSupported,
        c.NVML_ERROR_NO_PERMISSION => error.NoPermission,
        c.NVML_ERROR_ALREADY_INITIALIZED => error.AlreadyInitialized,
        c.NVML_ERROR_NOT_FOUND => error.NotFound,
        c.NVML_ERROR_INSUFFICIENT_SIZE => error.InsufficientSize,
        c.NVML_ERROR_INSUFFICIENT_POWER => error.InsufficientPower,
        c.NVML_ERROR_DRIVER_NOT_LOADED => error.DriverNotLoaded,
        c.NVML_ERROR_TIMEOUT => error.Timeout,
        c.NVML_ERROR_IRQ_ISSUE => error.IrqIssue,
        c.NVML_ERROR_LIBRARY_NOT_FOUND => error.LibraryNotFound,
        c.NVML_ERROR_FUNCTION_NOT_FOUND => error.FunctionNotFound,
        c.NVML_ERROR_CORRUPTED_INFOROM => error.CorruptedInforom,
        c.NVML_ERROR_GPU_IS_LOST => error.GpuIsLost,
        c.NVML_ERROR_RESET_REQUIRED => error.ResetRequired,
        c.NVML_ERROR_OPERATING_SYSTEM => error.OperatingSystem,
        c.NVML_ERROR_LIB_RM_VERSION_MISMATCH => error.LibRmVersionMismatch,
        c.NVML_ERROR_IN_USE => error.InUse,
        c.NVML_ERROR_MEMORY => error.Memory,
        c.NVML_ERROR_NO_DATA => error.NoData,
        c.NVML_ERROR_VGPU_ECC_NOT_SUPPORTED => error.VgpuEccNotSupported,
        c.NVML_ERROR_INSUFFICIENT_RESOURCES => error.InsufficientResources,
        c.NVML_ERROR_FREQ_NOT_SUPPORTED => error.FreqNotSupported,
        c.NVML_ERROR_ARGUMENT_VERSION_MISMATCH => error.ArgumentVersionMismatch,
        c.NVML_ERROR_DEPRECATED => error.Deprecated,
        c.NVML_ERROR_NOT_READY => error.NotReady,
        c.NVML_ERROR_GPU_NOT_FOUND => error.GpuNotFound,
        c.NVML_ERROR_INVALID_STATE => error.InvalidState,
        c.NVML_ERROR_RESET_TYPE_NOT_SUPPORTED => error.ResetTypeNotSupported,
        else => error.Unknown,
    };
}
