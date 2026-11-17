//! AMD GPUs, through the bundled libamd_smi, with the memory of each process from its fdinfo files.
const std = @import("std");
const builtin = @import("builtin");
const c = @import("c");
const contract = @import("zml-smi-2/contract");
const native = @import("zml-smi-2/native");
const polling = @import("zml-smi-2/poll");
const source = @import("zml-smi-2/source");

const AmdSmi = @This();
const log = std.log.scoped(.@"zml-smi-2/amdsmi");

devices: []const contract.Device,
io: std.Io,
api: Api,
gpus: []const Gpu,
proc: ?native.sysfs.Dir,
/// Graphics plus encode nanoseconds of each process on each GPU.
busy: native.Counters(struct { pid: u32, device: u16 }),

const Gpu = struct {
    handle: c.amdsmi_processor_handle,
    /// The PCI address fdinfo files name the GPU by: "0000:03:00.0".
    address: ?[12]u8,
};

pub const two_readings = true;

const Api = native.dynlib.Bindings(c, check, &.{
    "amdsmi_init",                 "amdsmi_status_code_to_string",
    "amdsmi_get_socket_handles",   "amdsmi_get_processor_handles",
    "amdsmi_get_gpu_asic_info",    "amdsmi_get_gpu_driver_info",
    "amdsmi_get_gpu_bdf_id",       "amdsmi_get_gpu_metrics_info",
    "amdsmi_get_power_info",       "amdsmi_get_temp_metric",
    "amdsmi_get_gpu_fan_speed",    "amdsmi_get_gpu_activity",
    "amdsmi_get_clock_info",       "amdsmi_get_gpu_memory_total",
    "amdsmi_get_gpu_memory_usage", "amdsmi_get_pcie_info",
    "amdsmi_get_gpu_process_list",
});

fn check(status: c.amdsmi_status_t) error{ Unsupported, AmdSmiError }!void {
    return switch (status) {
        c.AMDSMI_STATUS_SUCCESS => {},
        c.AMDSMI_STATUS_NOT_SUPPORTED, c.AMDSMI_STATUS_NOT_YET_IMPLEMENTED, c.AMDSMI_STATUS_NO_DATA => error.Unsupported,
        else => error.AmdSmiError,
    };
}

pub const detection_paths = [_][]const u8{"/dev/kfd"};

pub fn open(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !AmdSmi {
    const api = try native.dynlib.openBundled(Api, io, "libamd_smi.so.27");

    // The first call says why the library cannot be used.
    const status = api.amdsmi_init.function(c.AMDSMI_INIT_AMD_GPUS);
    check(status) catch |err| {
        const message: [*c]const u8 = api.amdsmi_status_code_to_string.get(.{status}) catch "unknown status";
        log.warn("amdsmi_init: {s}", .{std.mem.span(message)});
        return err;
    };

    var socket_count: u32 = 0;
    try api.amdsmi_get_socket_handles.call(.{ &socket_count, null });
    const sockets = try gpa.alloc(c.amdsmi_socket_handle, socket_count);
    defer gpa.free(sockets);
    try api.amdsmi_get_socket_handles.call(.{ &socket_count, sockets.ptr });

    var gpus: std.ArrayList(Gpu) = .empty;
    var devices: std.ArrayList(contract.Device) = .empty;
    for (sockets[0..socket_count]) |socket| {
        var count: u32 = 0;
        try api.amdsmi_get_processor_handles.call(.{ socket, &count, null });
        const handles = try gpa.alloc(c.amdsmi_processor_handle, count);
        defer gpa.free(handles);
        try api.amdsmi_get_processor_handles.call(.{ socket, &count, handles.ptr });

        for (handles[0..count]) |handle| {
            try gpus.append(arena, .{ .handle = handle, .address = if (api.amdsmi_get_gpu_bdf_id.get(.{handle})) |bdf| pciAddress(bdf) else |_| null });
            try devices.append(arena, .{
                .target = .rocm,
                .name = if (api.amdsmi_get_gpu_asic_info.get(.{handle})) |info| try arena.dupe(u8, std.mem.sliceTo(&info.market_name, 0)) else |_| null,
                .driver_version = if (api.amdsmi_get_gpu_driver_info.get(.{handle})) |info| try arena.dupe(u8, std.mem.sliceTo(&info.driver_version, 0)) else |_| null,
            });
        }
    }

    return .{
        .devices = devices.items,
        .io = io,
        .api = api,
        .gpus = gpus.items,
        .proc = native.sysfs.Dir.open(io, "/proc") catch null,
        .busy = .init(gpa),
    };
}

pub fn close(self: *AmdSmi) void {
    if (self.proc) |proc| proc.close();
    self.busy.deinit();
}

pub const poll = polling.PerDevice(@This()).poll;

pub const Probe = struct {
    api: *const Api,
    handle: c.amdsmi_processor_handle,

    const Scalar = native.dynlib.Scalar;
    pub const temperature = Scalar(Probe, .{
        .amdsmi_get_temp_metric, .{ c.AMDSMI_TEMPERATURE_TYPE_EDGE, c.AMDSMI_TEMP_CURRENT },
    });
    pub const mem_used_bytes =
        Scalar(Probe, .{ .amdsmi_get_gpu_memory_usage, .{c.AMDSMI_MEM_TYPE_VRAM} });
    pub const mem_total_bytes =
        Scalar(Probe, .{ .amdsmi_get_gpu_memory_total, .{c.AMDSMI_MEM_TYPE_VRAM} });

    pub fn power_mw(self: *const Probe) ?u64 {
        // Watts: the current power where the GPU has it, else the average.
        const metrics = self.api.amdsmi_get_gpu_metrics_info.get(.{self.handle}) catch return null;
        const watts = known(metrics.current_socket_power) orelse
            known(metrics.average_socket_power) orelse return null;
        return watts * 1000;
    }

    pub fn power_limit_mw(self: *const Probe) ?u64 {
        const info = self.api.amdsmi_get_power_info.get(.{self.handle}) catch return null;
        return native.units.scale(known(info.power_limit) orelse return null, .micro, .milli);
    }

    pub fn fan_speed_percent(self: *const Probe) ?u64 {
        const speed = self.api.amdsmi_get_gpu_fan_speed.get(.{ self.handle, 0 }) catch return null;
        return native.hwmon.pwmPercent(std.math.cast(u64, speed) orelse return null);
    }

    pub fn util_percent(self: *const Probe) ?u64 {
        const activity = self.api.amdsmi_get_gpu_activity.get(.{self.handle}) catch return null;
        // The API returns 0xFFFF when utilization is unavailable.
        return if (activity.gfx_activity <= 100) activity.gfx_activity else null;
    }

    const Clocks = contract.Reply(&.{
        .clock_graphics_mhz, .clock_graphics_max_mhz, .clock_soc_mhz,
        .clock_mem_mhz,      .clock_mem_max_mhz,
    });

    pub fn clocks(self: *const Probe) Clocks {
        const query = self.api.amdsmi_get_clock_info;
        const graphics = query.get(.{ self.handle, c.AMDSMI_CLK_TYPE_SYS }) catch null;
        const soc = query.get(.{ self.handle, c.AMDSMI_CLK_TYPE_SOC }) catch null;
        const memory = query.get(.{ self.handle, c.AMDSMI_CLK_TYPE_MEM }) catch null;
        return .{
            .clock_graphics_mhz = if (graphics) |clock| known(clock.clk) else null,
            .clock_graphics_max_mhz = if (graphics) |clock| known(clock.max_clk) else null,
            .clock_soc_mhz = if (soc) |clock| known(clock.clk) else null,
            .clock_mem_mhz = if (memory) |clock| known(clock.clk) else null,
            .clock_mem_max_mhz = if (memory) |clock| known(clock.max_clk) else null,
        };
    }

    pub fn pcie(self: *const Probe) contract.Reply(&.{
        .pcie_link_gen, .pcie_link_width, .pcie_bandwidth_mbps,
    }) {
        const data = self.api.amdsmi_get_pcie_info.get(.{self.handle}) catch return .{};
        const link: native.pci.Link = .{
            .gen = known(data.pcie_static.pcie_interface_version),
            .width = known(data.pcie_metric.pcie_width),
        };
        // Some GPUs report a fixed speed. Derive the rate from the link generation instead.
        return .{
            .pcie_link_gen = link.gen,
            .pcie_link_width = link.width,
            .pcie_bandwidth_mbps = link.mbps(),
        };
    }
};

pub fn probe(self: *AmdSmi, index: usize) Probe {
    return .{ .api = &self.api, .handle = self.gpus[index].handle };
}

pub fn readProcesses(self: *AmdSmi, scratch: std.mem.Allocator, reading: *source.Reading) !void {
    defer self.busy.sweep();
    var read_any = self.gpus.len == 0;

    for (self.gpus, 0..) |gpu, index| {
        const device: u16 = @intCast(index);
        const list = self.processList(scratch, gpu.handle) catch continue;
        read_any = true;
        for (list) |process| {
            const busy = try self.busy.advance(.{ .pid = process.pid, .device = device }, self.io, process.engine_usage.gfx +| process.engine_usage.enc);
            try reading.addProcess(.{
                .pid = process.pid,
                .device_idx = device,
                .dev_util_percent = if (busy) |growth| growth.percentOfTime() else null,
                .dev_mem_kib = if (gpu.address) |address| self.vramKib(process.pid, &address) else null,
            });
        }
    }
    if (!read_any) return error.Unavailable;
}

/// The processes on a GPU, from a query sized by a first call.
/// amdsmi prints while it lists processes, so stderr points at the null device for both calls.
fn processList(self: *AmdSmi, scratch: std.mem.Allocator, handle: c.amdsmi_processor_handle) ![]const c.amdsmi_proc_info_t {
    const saved = std.c.dup(std.c.STDERR_FILENO);
    if (saved >= 0) {
        const null_device = std.c.open("/dev/null", .{ .ACCMODE = .WRONLY });
        if (null_device >= 0) {
            _ = std.c.dup2(null_device, std.c.STDERR_FILENO);
            _ = std.c.close(null_device);
        }
    }
    defer if (saved >= 0) {
        _ = std.c.dup2(saved, std.c.STDERR_FILENO);
        _ = std.c.close(saved);
    };

    var count: u32 = 0;
    const query = self.api.amdsmi_get_gpu_process_list;
    try query.call(.{ handle, &count, null });
    if (count == 0) return &.{};

    const list = try scratch.alloc(c.amdsmi_proc_info_t, count);
    const status = query.function(handle, &count, list.ptr);
    // The list is full, and the processes that started since are left out.
    if (status != c.AMDSMI_STATUS_OUT_OF_RESOURCES) try check(status);
    return list[0..@min(count, list.len)];
}

/// The VRAM a process holds on a GPU, summed over its fdinfo files that name the GPU.
fn vramKib(self: *AmdSmi, pid: u32, address: []const u8) ?u64 {
    const proc = self.proc orelse return null;
    var path: [24]u8 = undefined;
    const files = proc.openDir(std.fmt.bufPrint(&path, "{d}/fdinfo", .{pid}) catch return null) catch return null;
    defer files.close();

    var bytes: u64 = 0;
    var buffer: [8192]u8 = undefined;
    var it = files.iterate();
    while (it.next(self.io) catch null) |entry| {
        const info = native.fdinfo.read(files, entry.name, &buffer) catch continue;
        if (!std.mem.eql(u8, info.pdev orelse continue, address)) continue;
        for (info.regions()) |region| {
            if (std.mem.eql(u8, region.name, "vram")) bytes += region.resident orelse 0;
        }
    }
    return bytes / 1024;
}

/// amdsmi reports a value it cannot read as the largest of its type.
fn known(value: anytype) ?u64 {
    return if (value == std.math.maxInt(@TypeOf(value))) null else value;
}

/// The PCI address of an amdsmi BDF id, whose domain is in the upper 32 bits.
fn pciAddress(bdf: u64) ?[12]u8 {
    var address: [12]u8 = undefined;
    _ = std.fmt.bufPrint(&address, "{x:0>4}:{x:0>2}:{x:0>2}.{x}", .{ bdf >> 32, (bdf >> 8) & 0xff, (bdf >> 3) & 0x1f, bdf & 0x7 }) catch return null;
    return address;
}

pub const testing = if (builtin.is_test) struct {
    pub const pciAddress = AmdSmi.pciAddress;
} else struct {};
