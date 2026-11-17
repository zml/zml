//! NVIDIA GPUs, through libnvidia-ml.
const std = @import("std");
const c = @import("c");
const model = @import("zml-smi/model");
const native = @import("zml-smi/native");
const source = @import("zml-smi/source");
const telemetry = @import("zml-smi/telemetry");

const Nvml = @This();
const log = std.log.scoped(.@"zml-smi/nvml");

devices: []const model.Device,
api: Api,
gpus: []Gpu,
tick: u8 = 0,

const Gpu = struct {
    handle: c.nvmlDevice_t,
    /// PCIe throughput is read every fourth poll and kept in between.
    pcie_tx_kbps: ?u64 = null,
    pcie_rx_kbps: ?u64 = null,
    /// The newest process utilization the GPU reported, so that a poll reads only newer ones.
    last_seen: u64 = 0,
};

const Api = struct {
    nvmlInit_v2: Checked(c.nvmlInit_v2),
    nvmlShutdown: *const @TypeOf(c.nvmlShutdown),
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

    const Checked = native.dynlib.CheckedBy(check);

    fn check(status: c.nvmlReturn_t) error{ Unsupported, NvmlError }!void {
        return switch (status) {
            c.NVML_SUCCESS => {},
            c.NVML_ERROR_NOT_SUPPORTED, c.NVML_ERROR_NO_DATA => error.Unsupported,
            else => error.NvmlError,
        };
    }
};

pub fn detect(io: std.Io) bool {
    return native.sysfs.exists(io, "/dev/nvidiactl");
}

pub fn open(arena: std.mem.Allocator, _: std.mem.Allocator, _: std.Io) !Nvml {
    const api = try native.dynlib.open(Api, &.{ "libnvidia-ml.so.1", "libnvidia-ml.so" });

    // The first call says why the library cannot be used.
    const status = api.nvmlInit_v2.function();
    Api.check(status) catch |err| {
        log.warn("nvmlInit: {s}", .{std.mem.span(api.nvmlErrorString(status))});
        return err;
    };

    errdefer _ = api.nvmlShutdown();

    var version: [c.NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE]u8 = undefined;
    const driver_version = if (api.nvmlSystemGetDriverVersion.call(.{ &version, version.len })) |_|
        try arena.dupe(u8, std.mem.sliceTo(&version, 0))
    else |_|
        null;
    const details: []const model.Device.Detail = if (api.nvmlSystemGetCudaDriverVersion_v2.get(.{})) |cuda| try arena.dupe(model.Device.Detail, &.{.{
        .key = "cuda_version",
        .label = "CUDA",
        .value = try std.fmt.allocPrint(arena, "{d}.{d}", .{ @divTrunc(cuda, 1000), @divTrunc(@mod(cuda, 1000), 10) }),
        .in_title = true,
    }}) else |_| &.{};

    const count = try api.nvmlDeviceGetCount_v2.get(.{});
    var gpus: std.ArrayList(Gpu) = try .initCapacity(arena, count);
    var devices: std.ArrayList(model.Device) = try .initCapacity(arena, count);
    for (0..count) |index| {
        // A GPU that fell off the bus is skipped; the others still show.
        const handle = api.nvmlDeviceGetHandleByIndex_v2.get(.{@intCast(index)}) catch continue;
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
        .devices = devices.items,
        .api = api,
        .gpus = gpus.items,
    };
}

pub fn close(self: *Nvml) void {
    _ = self.api.nvmlShutdown();
}

const pollGpu = telemetry.Bind(model.Gpu, .{
    .read = native.dynlib.readScalar,
    .fields = .{
        .power_mw = .{ .nvmlDeviceGetPowerUsage, .{} },
        .power_limit_mw = .{ .nvmlDeviceGetEnforcedPowerLimit, .{} },
        .temperature = .{ .nvmlDeviceGetTemperature, .{c.NVML_TEMPERATURE_GPU} },
        .fan_speed_percent = .{ .nvmlDeviceGetFanSpeed, .{} },
        .clock_graphics_mhz = .{ .nvmlDeviceGetClockInfo, .{c.NVML_CLOCK_GRAPHICS} },
        .clock_sm_mhz = .{ .nvmlDeviceGetClockInfo, .{c.NVML_CLOCK_SM} },
        .clock_mem_mhz = .{ .nvmlDeviceGetClockInfo, .{c.NVML_CLOCK_MEM} },
        .clock_graphics_max_mhz = .{ .nvmlDeviceGetMaxClockInfo, .{c.NVML_CLOCK_GRAPHICS} },
        .clock_mem_max_mhz = .{ .nvmlDeviceGetMaxClockInfo, .{c.NVML_CLOCK_MEM} },
        .mem_bus_width = .{ .nvmlDeviceGetMemoryBusWidth, .{} },
        .pcie_link_gen = .{ .nvmlDeviceGetCurrPcieLinkGeneration, .{} },
        .pcie_link_width = .{ .nvmlDeviceGetCurrPcieLinkWidth, .{} },
    },
    .query = readMemory,
}).poll;

fn readMemory(api: *const Api, handle: c.nvmlDevice_t) struct { mem_used_bytes: ?u64 = null, mem_total_bytes: ?u64 = null } {
    const memory = api.nvmlDeviceGetMemoryInfo.get(.{handle}) catch return .{};
    return .{ .mem_used_bytes = memory.used, .mem_total_bytes = memory.total };
}

pub fn poll(self: *Nvml, scratch: std.mem.Allocator, reading: *source.Reading) !void {
    const api = &self.api;
    // PCIe throughput blocks while it measures (#766).
    const read_pcie = self.tick % 4 == 0;
    self.tick +%= 1;

    for (self.gpus, reading.samples) |*gpu, *sample| {
        const fields = &sample.cuda;
        const handle = gpu.handle;
        pollGpu(fields, .{ api, handle });

        fields.util_percent = if (api.nvmlDeviceGetUtilizationRates.get(.{handle})) |rates| rates.gpu else |_| null;

        // Utilization and its sampling period.
        var util: c_uint, var period: c_uint = .{ 0, 0 };
        fields.encoder_util_percent = if (api.nvmlDeviceGetEncoderUtilization.call(.{ handle, &util, &period })) |_| util else |_| null;
        fields.decoder_util_percent = if (api.nvmlDeviceGetDecoderUtilization.call(.{ handle, &util, &period })) |_| util else |_| null;

        // NVML gives the speed of one lane.
        if (api.nvmlDeviceGetPcieSpeed.get(.{handle})) |lane_mbps| {
            if (fields.pcie_link_width) |width| fields.pcie_bandwidth_mbps = lane_mbps * width;
        } else |_| {}

        if (read_pcie) {
            gpu.pcie_tx_kbps = api.nvmlDeviceGetPcieThroughput.get(.{ handle, c.NVML_PCIE_UTIL_TX_BYTES }) catch null;
            gpu.pcie_rx_kbps = api.nvmlDeviceGetPcieThroughput.get(.{ handle, c.NVML_PCIE_UTIL_RX_BYTES }) catch null;
        }
        fields.pcie_tx_kbps = gpu.pcie_tx_kbps;
        fields.pcie_rx_kbps = gpu.pcie_rx_kbps;
    }

    // Finish every device's metrics before a process-list allocation can fail.
    var read_any = self.gpus.len == 0;

    for (self.gpus, 0..) |*gpu, device| {
        const handle = gpu.handle;
        const first_row = reading.processes.items.len;
        inline for (.{ api.nvmlDeviceGetComputeRunningProcesses_v3, api.nvmlDeviceGetGraphicsRunningProcesses_v3 }) |query| {
            if (running(scratch, query, handle)) |list| {
                read_any = true;
                for (list) |process| {
                    try reading.addProcess(.{ .pid = process.pid, .device_idx = @intCast(device), .dev_mem_kib = process.usedGpuMemory / 1024 });
                }
            } else |_| {}
        }

        const since = gpu.last_seen;
        const utilization = self.processUtilization(scratch, handle, since) catch continue;
        for (utilization) |util| {
            if (util.smUtil > 100 or util.timeStamp <= since) continue;
            gpu.last_seen = @max(gpu.last_seen, util.timeStamp);
            for (reading.processes.items[first_row..]) |*row| {
                if (row.pid == util.pid) row.dev_util_percent = @intCast(util.smUtil);
            }
        }
    }
    if (!read_any) return error.Unavailable;
}

/// The processes of a GPU, from a query sized by a first call.
fn running(scratch: std.mem.Allocator, query: anytype, handle: c.nvmlDevice_t) ![]const c.nvmlProcessInfo_t {
    var count: c_uint = 0;
    const status = query.function(handle, &count, null);
    if (status == c.NVML_SUCCESS) return &.{};
    if (status != c.NVML_ERROR_INSUFFICIENT_SIZE) try Api.check(status);

    const list = try scratch.alloc(c.nvmlProcessInfo_t, count);
    try query.call(.{ handle, &count, list.ptr });
    return list[0..count];
}

/// The utilization samples newer than `since`, from a query sized by a first call.
fn processUtilization(self: *Nvml, scratch: std.mem.Allocator, handle: c.nvmlDevice_t, since: u64) ![]const c.nvmlProcessUtilizationSample_t {
    var count: c_uint = 0;
    const status = self.api.nvmlDeviceGetProcessUtilization.function(handle, null, &count, since);
    if (status == c.NVML_SUCCESS) return &.{};
    if (status != c.NVML_ERROR_INSUFFICIENT_SIZE) try Api.check(status);

    const samples = try scratch.alloc(c.nvmlProcessUtilizationSample_t, count);
    try self.api.nvmlDeviceGetProcessUtilization.call(.{ handle, samples.ptr, &count, since });
    return samples[0..count];
}
