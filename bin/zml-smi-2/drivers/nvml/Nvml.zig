//! NVIDIA GPUs, through libnvidia-ml.
const std = @import("std");
const c = @import("c");
const contract = @import("zml-smi-2/contract");
const native = @import("zml-smi-2/native");
const polling = @import("zml-smi-2/poll");
const source = @import("zml-smi-2/source");

const Nvml = @This();
const log = std.log.scoped(.@"zml-smi-2/nvml");

devices: []const contract.Device,
api: Api,
gpus: []Gpu,
/// The first poll wraps to phase zero and reads PCIe throughput.
phase: u2 = 3,

const Throughput = contract.Reply(&.{ .pcie_tx_kbps, .pcie_rx_kbps });

const Gpu = struct {
    handle: c.nvmlDevice_t,
    /// PCIe throughput is read every fourth poll and kept in between.
    throughput: Throughput = .{},
    /// The newest process utilization the GPU reported, so that a poll reads only newer ones.
    last_seen: u64 = 0,
};

const scalar_bindings = .{
    .nvmlDeviceGetPowerUsage = .{ .power_mw = .{} },
    .nvmlDeviceGetEnforcedPowerLimit = .{ .power_limit_mw = .{} },
    .nvmlDeviceGetTemperature = .{ .temperature = .{c.NVML_TEMPERATURE_GPU} },
    .nvmlDeviceGetFanSpeed = .{ .fan_speed_percent = .{} },
    .nvmlDeviceGetClockInfo = .{
        .clock_graphics_mhz = .{c.NVML_CLOCK_GRAPHICS},
        .clock_sm_mhz = .{c.NVML_CLOCK_SM},
        .clock_mem_mhz = .{c.NVML_CLOCK_MEM},
    },
    .nvmlDeviceGetMaxClockInfo = .{
        .clock_graphics_max_mhz = .{c.NVML_CLOCK_GRAPHICS},
        .clock_mem_max_mhz = .{c.NVML_CLOCK_MEM},
    },
    .nvmlDeviceGetMemoryBusWidth = .{ .mem_bus_width = .{} },
    .nvmlDeviceGetCurrPcieLinkGeneration = .{ .pcie_link_gen = .{} },
};

const Api = native.dynlib.Bindings(c, check, .{
    "nvmlInit_v2",                              "nvmlShutdown",
    "nvmlErrorString",                          "nvmlSystemGetDriverVersion",
    "nvmlSystemGetCudaDriverVersion_v2",        "nvmlDeviceGetCount_v2",
    "nvmlDeviceGetHandleByIndex_v2",            "nvmlDeviceGetName",
    "nvmlDeviceGetUtilizationRates",            "nvmlDeviceGetEncoderUtilization",
    "nvmlDeviceGetDecoderUtilization",          "nvmlDeviceGetMemoryInfo",
    "nvmlDeviceGetPcieThroughput",              "nvmlDeviceGetPcieSpeed",
    "nvmlDeviceGetCurrPcieLinkWidth",           "nvmlDeviceGetComputeRunningProcesses_v3",
    "nvmlDeviceGetGraphicsRunningProcesses_v3", "nvmlDeviceGetProcessUtilization",
} ++ @typeInfo(@TypeOf(scalar_bindings)).@"struct".field_names);

fn check(status: c.nvmlReturn_t) error{ Unsupported, NvmlError }!void {
    return switch (status) {
        c.NVML_SUCCESS => {},
        c.NVML_ERROR_NOT_SUPPORTED, c.NVML_ERROR_NO_DATA => error.Unsupported,
        else => error.NvmlError,
    };
}

pub const detection_paths = [_][]const u8{"/dev/nvidiactl"};

pub fn open(arena: std.mem.Allocator, _: std.mem.Allocator, _: std.Io) !Nvml {
    const api = try native.dynlib.open(Api, &.{ "libnvidia-ml.so.1", "libnvidia-ml.so" });

    // The first call says why the library cannot be used.
    const status = api.nvmlInit_v2.function();
    check(status) catch |err| {
        log.warn("nvmlInit: {s}", .{std.mem.span(api.nvmlErrorString.function(status))});
        return err;
    };

    errdefer _ = api.nvmlShutdown.function();

    var version: [c.NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE]u8 = undefined;
    const driver_version = if (api.nvmlSystemGetDriverVersion.call(.{ &version, version.len })) |_|
        try arena.dupe(u8, std.mem.sliceTo(&version, 0))
    else |_|
        null;
    const details: []const contract.Device.Detail = if (api.nvmlSystemGetCudaDriverVersion_v2.get(.{})) |cuda| try arena.dupe(contract.Device.Detail, &.{.{
        .key = "cuda_version",
        .value = try std.fmt.allocPrint(arena, "{d}.{d}", .{ @divTrunc(cuda, 1000), @divTrunc(@mod(cuda, 1000), 10) }),
    }}) else |_| &.{};

    const count = try api.nvmlDeviceGetCount_v2.get(.{});
    var gpus: std.ArrayList(Gpu) = try .initCapacity(arena, count);
    var devices: std.ArrayList(contract.Device) = try .initCapacity(arena, count);
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
    _ = self.api.nvmlShutdown.function();
}

pub const poll = polling.perDevice;

pub const Probe = struct {
    api: *const Api,
    handle: c.nvmlDevice_t,
    gpu: *Gpu,
    refresh_pcie: bool,

    pub const scalars = native.dynlib.Scalars(Probe, scalar_bindings).read;

    pub fn memory(self: *const Probe) contract.Reply(&.{ .mem_used_bytes, .mem_total_bytes }) {
        const data = self.api.nvmlDeviceGetMemoryInfo.get(.{self.handle}) catch return .{};
        return .{ .mem_used_bytes = data.used, .mem_total_bytes = data.total };
    }

    pub fn util_percent(self: *const Probe) ?u64 {
        const rates = self.api.nvmlDeviceGetUtilizationRates.get(.{self.handle}) catch return null;
        return rates.gpu;
    }

    pub const encoder_util_percent = engineUtilization(.nvmlDeviceGetEncoderUtilization);
    pub const decoder_util_percent = engineUtilization(.nvmlDeviceGetDecoderUtilization);

    pub fn pcie(self: *const Probe) contract.Reply(&.{ .pcie_link_width, .pcie_bandwidth_mbps }) {
        const width: ?u64 = self.api.nvmlDeviceGetCurrPcieLinkWidth.get(.{self.handle}) catch null;
        const speed: ?u64 = self.api.nvmlDeviceGetPcieSpeed.get(.{self.handle}) catch null;
        return .{
            .pcie_link_width = width,
            .pcie_bandwidth_mbps = if (width != null and speed != null) width.? * speed.? else null,
        };
    }

    pub fn pcieThroughput(self: *const Probe) Throughput {
        // Throughput blocks while it measures (#766), so keep it between refreshes.
        if (self.refresh_pcie) {
            const query = self.api.nvmlDeviceGetPcieThroughput;
            self.gpu.throughput = .{
                .pcie_tx_kbps = query.get(.{ self.handle, c.NVML_PCIE_UTIL_TX_BYTES }) catch null,
                .pcie_rx_kbps = query.get(.{ self.handle, c.NVML_PCIE_UTIL_RX_BYTES }) catch null,
            };
        }
        return self.gpu.throughput;
    }

    fn engineUtilization(comptime function: anytype) fn (*const Probe) ?u64 {
        return struct {
            fn read(self: *const Probe) ?u64 {
                const query = @field(self.api, @tagName(function));
                var util: c_uint, var period: c_uint = .{ 0, 0 };
                query.call(.{ self.handle, &util, &period }) catch return null;
                return util;
            }
        }.read;
    }
};

pub fn begin(self: *Nvml) void {
    self.phase +%= 1;
}

pub fn probe(self: *Nvml, index: usize) Probe {
    const gpu = &self.gpus[index];
    return .{
        .api = &self.api,
        .handle = gpu.handle,
        .gpu = gpu,
        .refresh_pcie = self.phase == 0,
    };
}

pub fn readProcesses(self: *Nvml, scratch: std.mem.Allocator, reading: *source.Reading) !void {
    // Finish every device's metrics before a process-list allocation can fail.
    const api = &self.api;
    var read_any = self.gpus.len == 0;

    for (self.gpus, 0..) |*gpu, device| {
        const handle = gpu.handle;
        const first_row = reading.processes.items.len;
        for ([_]@TypeOf(api.nvmlDeviceGetComputeRunningProcesses_v3){
            api.nvmlDeviceGetComputeRunningProcesses_v3,
            api.nvmlDeviceGetGraphicsRunningProcesses_v3,
        }) |query| {
            const list = readList(c.nvmlProcessInfo_t, scratch, query, handle, {}) catch continue;
            read_any = true;
            for (list) |process| {
                try reading.addProcess(.{
                    .pid = process.pid,
                    .device_idx = @intCast(device),
                    .dev_mem_kib = process.usedGpuMemory / 1024,
                });
            }
        }

        const since = gpu.last_seen;
        const query = api.nvmlDeviceGetProcessUtilization;
        const Sample = c.nvmlProcessUtilizationSample_t;
        const usage = readList(Sample, scratch, query, handle, since) catch continue;
        for (usage) |util| {
            if (util.smUtil > 100 or util.timeStamp <= since) continue;
            gpu.last_seen = @max(gpu.last_seen, util.timeStamp);
            for (reading.processes.items[first_row..]) |*row| {
                if (row.pid == util.pid) row.dev_util_percent = @intCast(util.smUtil);
            }
        }
    }
    if (!read_any) return error.Unavailable;
}

/// A null output first sizes both kinds of NVML process table. The utilization API
/// also takes a timestamp and puts the count after the output pointer.
fn readList(
    comptime Item: type,
    scratch: std.mem.Allocator,
    query: anytype,
    handle: c.nvmlDevice_t,
    since: if (Item == c.nvmlProcessInfo_t) void else u64,
) ![]const Item {
    const running = Item == c.nvmlProcessInfo_t;
    var count: c_uint = 0;
    var args: @TypeOf(query).Args = if (running)
        .{ handle, &count, null }
    else
        .{ handle, null, &count, since };
    const status = @call(.auto, query.function, args);
    if (status == c.NVML_SUCCESS) return &.{};
    if (status != c.NVML_ERROR_INSUFFICIENT_SIZE) try check(status);
    const list = try scratch.alloc(Item, count);
    args[if (running) 2 else 1] = list.ptr;
    try query.call(args);
    return list[0..count];
}
