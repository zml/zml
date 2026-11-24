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
    nvmlInit_v2: Checked(c.nvmlInit_v2),
    nvmlErrorString: *const @TypeOf(c.nvmlErrorString),
    nvmlSystemGetDriverVersion: Checked(c.nvmlSystemGetDriverVersion),
    nvmlSystemGetCudaDriverVersion_v2: Checked(c.nvmlSystemGetCudaDriverVersion_v2),
    nvmlDeviceGetCount_v2: Checked(c.nvmlDeviceGetCount_v2),
    nvmlDeviceGetHandleByIndex_v2: Checked(c.nvmlDeviceGetHandleByIndex_v2),
    nvmlDeviceGetName: Checked(c.nvmlDeviceGetName),
    nvmlDeviceGetPowerUsage: Checked(c.nvmlDeviceGetPowerUsage),
    nvmlDeviceGetEnforcedPowerLimit: Checked(c.nvmlDeviceGetEnforcedPowerLimit),
    nvmlDeviceGetTemperature: Checked(c.nvmlDeviceGetTemperature),
    nvmlDeviceGetFanSpeed: Checked(c.nvmlDeviceGetFanSpeed),
    nvmlDeviceGetUtilizationRates: Checked(c.nvmlDeviceGetUtilizationRates),
    nvmlDeviceGetEncoderUtilization: Checked(c.nvmlDeviceGetEncoderUtilization),
    nvmlDeviceGetDecoderUtilization: Checked(c.nvmlDeviceGetDecoderUtilization),
    nvmlDeviceGetClockInfo: Checked(c.nvmlDeviceGetClockInfo),
    nvmlDeviceGetMaxClockInfo: Checked(c.nvmlDeviceGetMaxClockInfo),
    nvmlDeviceGetMemoryInfo: Checked(c.nvmlDeviceGetMemoryInfo),
    nvmlDeviceGetMemoryBusWidth: Checked(c.nvmlDeviceGetMemoryBusWidth),
    nvmlDeviceGetPcieThroughput: Checked(c.nvmlDeviceGetPcieThroughput),
    nvmlDeviceGetPcieSpeed: Checked(c.nvmlDeviceGetPcieSpeed),
    nvmlDeviceGetCurrPcieLinkGeneration: Checked(c.nvmlDeviceGetCurrPcieLinkGeneration),
    nvmlDeviceGetCurrPcieLinkWidth: Checked(c.nvmlDeviceGetCurrPcieLinkWidth),
    nvmlDeviceGetComputeRunningProcesses_v3: Checked(c.nvmlDeviceGetComputeRunningProcesses_v3),
    nvmlDeviceGetGraphicsRunningProcesses_v3: Checked(c.nvmlDeviceGetGraphicsRunningProcesses_v3),
    nvmlDeviceGetProcessUtilization: Checked(c.nvmlDeviceGetProcessUtilization),

    fn Checked(comptime function: anytype) type {
        return smi.dynlib.Checked(@TypeOf(function), check);
    }

    fn check(status: c.nvmlReturn_t) error{ Unsupported, NvmlError }!void {
        return switch (status) {
            c.NVML_SUCCESS => {},
            c.NVML_ERROR_NOT_SUPPORTED, c.NVML_ERROR_NO_DATA => error.Unsupported,
            else => error.NvmlError,
        };
    }
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

    const status = api.nvmlInit_v2.function();
    Api.check(status) catch |err| {
        log.warn("nvmlInit: {s}", .{std.mem.span(api.nvmlErrorString(status))});
        return err;
    };

    var version: [c.NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE]u8 = undefined;
    const driver_version = if (api.nvmlSystemGetDriverVersion.call(.{ &version, version.len })) |_|
        try arena.dupe(u8, std.mem.sliceTo(&version, 0))
    else |_|
        null;
    var cuda: c_int = 0;
    const details: []const smi.Device.Detail = if (api.nvmlSystemGetCudaDriverVersion_v2.call(.{&cuda})) |_| try arena.dupe(smi.Device.Detail, &.{.{
        .key = "cuda_version",
        .label = "CUDA",
        .value = try std.fmt.allocPrint(arena, "{d}.{d}", .{ @divTrunc(cuda, 1000), @divTrunc(@mod(cuda, 1000), 10) }),
        .in_title = true,
    }}) else |_| &.{};

    var count: c_uint = 0;
    try api.nvmlDeviceGetCount_v2.call(.{&count});
    var gpus: std.ArrayList(Gpu) = try .initCapacity(arena, count);
    var devices: std.ArrayList(smi.Device) = try .initCapacity(arena, count);
    for (0..count) |index| {
        var handle: c.nvmlDevice_t = undefined;
        api.nvmlDeviceGetHandleByIndex_v2.call(.{ @intCast(index), &handle }) catch continue;
        var name: [c.NVML_DEVICE_NAME_V2_BUFFER_SIZE]u8 = undefined;
        gpus.appendAssumeCapacity(.{ .handle = handle });
        devices.appendAssumeCapacity(.{
            .target = .cuda,
            .name = if (api.nvmlDeviceGetName.call(.{ handle, &name, name.len })) |_| try arena.dupe(u8, std.mem.sliceTo(&name, 0)) else |_| null,
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
    // PCIe throughput blocks while it measures (#766).
    const read_pcie = self.tick % 4 == 0;
    self.tick +%= 1;

    for (self.gpus, samples) |*gpu, *sample| {
        const fields = &sample.cuda;
        const handle = gpu.handle;

        var value: c_uint = 0;
        fields.power_mw = if (api.nvmlDeviceGetPowerUsage.call(.{ handle, &value })) |_| value else |_| null;
        fields.power_limit_mw = if (api.nvmlDeviceGetEnforcedPowerLimit.call(.{ handle, &value })) |_| value else |_| null;
        fields.temperature = if (api.nvmlDeviceGetTemperature.call(.{ handle, c.NVML_TEMPERATURE_GPU, &value })) |_| value else |_| null;
        fields.fan_speed_percent = if (api.nvmlDeviceGetFanSpeed.call(.{ handle, &value })) |_| value else |_| null;
        var rates: c.nvmlUtilization_t = undefined;
        fields.util_percent = if (api.nvmlDeviceGetUtilizationRates.call(.{ handle, &rates })) |_| rates.gpu else |_| null;

        var util: c_uint, var period: c_uint = .{ 0, 0 };
        fields.encoder_util_percent = if (api.nvmlDeviceGetEncoderUtilization.call(.{ handle, &util, &period })) |_| util else |_| null;
        fields.decoder_util_percent = if (api.nvmlDeviceGetDecoderUtilization.call(.{ handle, &util, &period })) |_| util else |_| null;

        fields.clock_graphics_mhz = if (api.nvmlDeviceGetClockInfo.call(.{ handle, c.NVML_CLOCK_GRAPHICS, &value })) |_| value else |_| null;
        fields.clock_sm_mhz = if (api.nvmlDeviceGetClockInfo.call(.{ handle, c.NVML_CLOCK_SM, &value })) |_| value else |_| null;
        fields.clock_mem_mhz = if (api.nvmlDeviceGetClockInfo.call(.{ handle, c.NVML_CLOCK_MEM, &value })) |_| value else |_| null;
        fields.clock_graphics_max_mhz = if (api.nvmlDeviceGetMaxClockInfo.call(.{ handle, c.NVML_CLOCK_GRAPHICS, &value })) |_| value else |_| null;
        fields.clock_mem_max_mhz = if (api.nvmlDeviceGetMaxClockInfo.call(.{ handle, c.NVML_CLOCK_MEM, &value })) |_| value else |_| null;

        var memory: c.nvmlMemory_t = undefined;
        if (api.nvmlDeviceGetMemoryInfo.call(.{ handle, &memory })) |_| {
            fields.mem_used_bytes = memory.used;
            fields.mem_total_bytes = memory.total;
        } else |_| {}
        fields.mem_bus_width = if (api.nvmlDeviceGetMemoryBusWidth.call(.{ handle, &value })) |_| value else |_| null;

        fields.pcie_link_gen = if (api.nvmlDeviceGetCurrPcieLinkGeneration.call(.{ handle, &value })) |_| value else |_| null;
        fields.pcie_link_width = if (api.nvmlDeviceGetCurrPcieLinkWidth.call(.{ handle, &value })) |_| value else |_| null;
        // NVML gives the speed of one lane.
        if (api.nvmlDeviceGetPcieSpeed.call(.{ handle, &value })) |_| {
            if (fields.pcie_link_width) |width| fields.pcie_bandwidth_mbps = value * width;
        } else |_| {}

        if (read_pcie) {
            gpu.pcie_tx_kbps = if (api.nvmlDeviceGetPcieThroughput.call(.{ handle, c.NVML_PCIE_UTIL_TX_BYTES, &value })) |_| value else |_| null;
            gpu.pcie_rx_kbps = if (api.nvmlDeviceGetPcieThroughput.call(.{ handle, c.NVML_PCIE_UTIL_RX_BYTES, &value })) |_| value else |_| null;
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
    const status = query.function(handle, &count, null);
    if (status == c.NVML_SUCCESS) return &.{};
    if (status != c.NVML_ERROR_INSUFFICIENT_SIZE) try Api.check(status);

    const list = try scratch.alloc(c.nvmlProcessInfo_t, count);
    try query.call(.{ handle, &count, list.ptr });
    return list[0..count];
}

fn processUtilization(self: *Nvml, scratch: std.mem.Allocator, handle: c.nvmlDevice_t, since: u64) ![]const c.nvmlProcessUtilizationSample_t {
    var count: c_uint = 0;
    const status = self.api.nvmlDeviceGetProcessUtilization.function(handle, null, &count, since);
    if (status == c.NVML_SUCCESS) return &.{};
    if (status != c.NVML_ERROR_INSUFFICIENT_SIZE) try Api.check(status);

    const samples = try scratch.alloc(c.nvmlProcessUtilizationSample_t, count);
    try self.api.nvmlDeviceGetProcessUtilization.call(.{ handle, samples.ptr, &count, since });
    return samples[0..count];
}
