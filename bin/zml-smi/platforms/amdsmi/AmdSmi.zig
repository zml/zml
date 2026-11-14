//! AMD GPUs, through the bundled libamd_smi, with the memory of each process from its fdinfo files.
const std = @import("std");
const c = @import("c");
const smi = @import("zml-smi");

const AmdSmi = @This();
const log = std.log.scoped(.@"zml-smi/amdsmi");

platform: smi.Platform,
io: std.Io,
api: Api,
gpus: []const Gpu,
proc: ?smi.sysfs.Dir,
/// Graphics plus encode nanoseconds of each process on each GPU.
busy: smi.Counters(struct { pid: u32, device: u16 }),

const Gpu = struct {
    handle: c.amdsmi_processor_handle,
    /// The PCI address fdinfo files name the GPU by: "0000:03:00.0".
    address: ?[12]u8,
};

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes, .deinit = deinit };

const Api = struct {
    amdsmi_init: Checked(c.amdsmi_init),
    amdsmi_status_code_to_string: Checked(c.amdsmi_status_code_to_string),
    amdsmi_get_socket_handles: Checked(c.amdsmi_get_socket_handles),
    amdsmi_get_processor_handles: Checked(c.amdsmi_get_processor_handles),
    amdsmi_get_gpu_asic_info: Checked(c.amdsmi_get_gpu_asic_info),
    amdsmi_get_gpu_driver_info: Checked(c.amdsmi_get_gpu_driver_info),
    amdsmi_get_gpu_bdf_id: Checked(c.amdsmi_get_gpu_bdf_id),
    amdsmi_get_gpu_metrics_info: Checked(c.amdsmi_get_gpu_metrics_info),
    amdsmi_get_power_info: Checked(c.amdsmi_get_power_info),
    amdsmi_get_temp_metric: Checked(c.amdsmi_get_temp_metric),
    amdsmi_get_gpu_fan_speed: Checked(c.amdsmi_get_gpu_fan_speed),
    amdsmi_get_gpu_activity: Checked(c.amdsmi_get_gpu_activity),
    amdsmi_get_clock_info: Checked(c.amdsmi_get_clock_info),
    amdsmi_get_gpu_memory_total: Checked(c.amdsmi_get_gpu_memory_total),
    amdsmi_get_gpu_memory_usage: Checked(c.amdsmi_get_gpu_memory_usage),
    amdsmi_get_pcie_info: Checked(c.amdsmi_get_pcie_info),
    amdsmi_get_gpu_process_list: Checked(c.amdsmi_get_gpu_process_list),

    fn Checked(comptime function: anytype) type {
        return smi.dynlib.Checked(function, check);
    }

    fn check(status: c.amdsmi_status_t) error{ Unsupported, AmdSmiError }!void {
        return switch (status) {
            c.AMDSMI_STATUS_SUCCESS => {},
            c.AMDSMI_STATUS_NOT_SUPPORTED, c.AMDSMI_STATUS_NOT_YET_IMPLEMENTED, c.AMDSMI_STATUS_NO_DATA => error.Unsupported,
            else => error.AmdSmiError,
        };
    }
};

pub fn detect(io: std.Io) bool {
    return smi.sysfs.exists(io, "/dev/kfd");
}

pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !AmdSmi {
    const api = try smi.dynlib.openBundled(Api, io, "libamd_smi.so.27");

    // The first call says why the library cannot be used.
    const status = api.amdsmi_init.function(c.AMDSMI_INIT_AMD_GPUS);
    Api.check(status) catch |err| {
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
    var devices: std.ArrayList(smi.Device) = .empty;
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
        .platform = .{ .devices = devices.items, .two_readings = true, .vtable = &vtable },
        .io = io,
        .api = api,
        .gpus = gpus.items,
        .proc = smi.sysfs.Dir.open(io, "/proc") catch null,
        .busy = .init(gpa),
    };
}

fn deinit(platform: *smi.Platform) void {
    const self: *AmdSmi = @alignCast(@fieldParentPtr("platform", platform));
    if (self.proc) |proc| proc.close();
    self.busy.deinit();
}

fn poll(platform: *smi.Platform, _: std.mem.Allocator, samples: []smi.Sample) void {
    const self: *AmdSmi = @alignCast(@fieldParentPtr("platform", platform));
    const api = &self.api;

    for (self.gpus, samples) |gpu, *sample| {
        const fields = &sample.rocm;
        const handle = gpu.handle;

        // Watts: the current power where the GPU has it, else the average.
        if (api.amdsmi_get_gpu_metrics_info.get(.{handle})) |metrics| {
            if (known(metrics.current_socket_power) orelse known(metrics.average_socket_power)) |watts| fields.power_mw = watts * 1000;
        } else |_| {}
        if (api.amdsmi_get_power_info.get(.{handle})) |info| {
            if (known(info.power_limit)) |microwatts| fields.power_limit_mw = smi.units.scale(microwatts, .micro, .milli);
        } else |_| {}
        if (api.amdsmi_get_temp_metric.get(.{ handle, c.AMDSMI_TEMPERATURE_TYPE_EDGE, c.AMDSMI_TEMP_CURRENT })) |celsius| {
            fields.temperature = std.math.cast(u64, celsius);
        } else |_| {}
        if (api.amdsmi_get_gpu_fan_speed.get(.{ handle, 0 })) |speed| {
            if (std.math.cast(u64, speed)) |pwm| fields.fan_speed_percent = smi.hwmon.pwmPercent(pwm);
        } else |_| {}
        if (api.amdsmi_get_gpu_activity.get(.{handle})) |activity| {
            // Not a percentage when unsupported: 0xFFFF.
            if (activity.gfx_activity <= 100) fields.util_percent = activity.gfx_activity;
        } else |_| {}

        if (api.amdsmi_get_clock_info.get(.{ handle, c.AMDSMI_CLK_TYPE_SYS })) |clock| {
            fields.clock_graphics_mhz = known(clock.clk);
            fields.clock_graphics_max_mhz = known(clock.max_clk);
        } else |_| {}
        if (api.amdsmi_get_clock_info.get(.{ handle, c.AMDSMI_CLK_TYPE_SOC })) |clock| {
            fields.clock_soc_mhz = known(clock.clk);
        } else |_| {}
        if (api.amdsmi_get_clock_info.get(.{ handle, c.AMDSMI_CLK_TYPE_MEM })) |clock| {
            fields.clock_mem_mhz = known(clock.clk);
            fields.clock_mem_max_mhz = known(clock.max_clk);
        } else |_| {}

        fields.mem_used_bytes = api.amdsmi_get_gpu_memory_usage.get(.{ handle, c.AMDSMI_MEM_TYPE_VRAM }) catch null;
        fields.mem_total_bytes = api.amdsmi_get_gpu_memory_total.get(.{ handle, c.AMDSMI_MEM_TYPE_VRAM }) catch null;

        if (api.amdsmi_get_pcie_info.get(.{handle})) |pcie| {
            const link: smi.pci.Link = .{ .gen = known(pcie.pcie_static.pcie_interface_version), .width = known(pcie.pcie_metric.pcie_width) };
            fields.pcie_link_gen = link.gen;
            fields.pcie_link_width = link.width;
            // The rate comes from the generation: the speed amdsmi reports stays at 2.5 GT/s on an R9700 whose link runs at 32.
            fields.pcie_bandwidth_mbps = link.mbps();
        } else |_| {}
    }
}

fn processes(platform: *smi.Platform, scratch: std.mem.Allocator, _: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
    const self: *AmdSmi = @alignCast(@fieldParentPtr("platform", platform));
    defer self.busy.sweep();
    var read_any = self.gpus.len == 0;

    for (self.gpus, 0..) |gpu, index| {
        const device: u16 = @intCast(index);
        const list = self.processList(scratch, gpu.handle) catch continue;
        read_any = true;
        for (list) |process| {
            const busy = try self.busy.advance(.{ .pid = process.pid, .device = device }, self.io, process.engine_usage.gfx +| process.engine_usage.enc);
            try out.append(.{
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
fn processList(self: *AmdSmi, scratch: std.mem.Allocator, handle: c.amdsmi_processor_handle) ![]const c.amdsmi_proc_info_t {
    var count: u32 = 0;
    try Api.check(self.listQuietly(handle, &count, null));
    if (count == 0) return &.{};

    const list = try scratch.alloc(c.amdsmi_proc_info_t, count);
    const status = self.listQuietly(handle, &count, list.ptr);
    // The list is full, and the processes that started since are left out.
    if (status != c.AMDSMI_STATUS_OUT_OF_RESOURCES) try Api.check(status);
    return list[0..@min(count, list.len)];
}

/// amdsmi prints while it lists processes, so stderr points at the null device for the length of the call.
fn listQuietly(self: *AmdSmi, handle: c.amdsmi_processor_handle, count: *u32, list: [*c]c.amdsmi_proc_info_t) c.amdsmi_status_t {
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
    return self.api.amdsmi_get_gpu_process_list.function(handle, count, list);
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
        const info = smi.fdinfo.read(files, entry.name, &buffer) catch continue;
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

test pciAddress {
    try std.testing.expectEqualStrings("0000:03:00.0", &pciAddress(0x0300).?);
    // Bits 28 to 31 hold a partition id, which is not part of the address.
    try std.testing.expectEqualStrings("0001:c1:00.1", &pciAddress(0x1_1000_c101).?);
}
