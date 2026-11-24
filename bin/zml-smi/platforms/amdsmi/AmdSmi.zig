const std = @import("std");
const c = @import("c");
const smi = @import("zml-smi");

const AmdSmi = @This();
const log = std.log.scoped(.@"zml-smi/amdsmi");

platform: smi.Platform,
io: std.Io,
gpa: std.mem.Allocator,
api: Api,
gpus: []const Gpu,
prev_busy: std.AutoHashMapUnmanaged(BusyKey, std.Io.Duration) = .empty,
curr_busy: std.AutoHashMapUnmanaged(BusyKey, std.Io.Duration) = .empty,
last_poll: ?std.Io.Timestamp = null,

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
        return smi.dynlib.Checked(@TypeOf(function), check);
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
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox = smi.sandbox.path(gpa, io, &buffer) orelse return error.LibraryNotFound;
    var path: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const library = std.c.dlopen(try std.fmt.bufPrintSentinel(&path, "{s}/lib/libamd_smi.so.27", .{sandbox}, 0), .{ .LAZY = true, .GLOBAL = true, .NODELETE = true }) orelse {
        if (std.c.dlerror()) |err| log.warn("dlopen: {s}", .{err});
        return error.LibraryNotFound;
    };
    const api = try smi.dynlib.lookupStruct(library, Api);

    const status = api.amdsmi_init.function(c.AMDSMI_INIT_AMD_GPUS);
    Api.check(status) catch |err| {
        var message: [*c]const u8 = "unknown status";
        api.amdsmi_status_code_to_string.call(.{ status, &message }) catch {};
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
            var bdf: u64 = 0;
            var asic: c.amdsmi_asic_info_t = undefined;
            var driver: c.amdsmi_driver_info_t = undefined;
            try gpus.append(arena, .{ .handle = handle, .address = if (api.amdsmi_get_gpu_bdf_id.call(.{ handle, &bdf })) |_| pciAddress(bdf) else |_| null });
            // amdsmi reports the version of an in-tree driver, which has none, as N/A.
            const version = if (api.amdsmi_get_gpu_driver_info.call(.{ handle, &driver })) |_| std.mem.sliceTo(&driver.driver_version, 0) else |_| "N/A";
            try devices.append(arena, .{
                .target = .rocm,
                .name = if (api.amdsmi_get_gpu_asic_info.call(.{ handle, &asic })) |_| try arena.dupe(u8, std.mem.sliceTo(&asic.market_name, 0)) else |_| null,
                .driver_version = if (std.mem.eql(u8, version, "N/A")) null else try arena.dupe(u8, version),
            });
        }
    }

    return .{
        .platform = .{ .devices = devices.items, .two_readings = true, .vtable = &vtable },
        .io = io,
        .gpa = gpa,
        .api = api,
        .gpus = gpus.items,
    };
}

fn poll(platform: *smi.Platform, _: std.mem.Allocator, samples: []smi.Sample) void {
    const self: *AmdSmi = @alignCast(@fieldParentPtr("platform", platform));
    const api = &self.api;

    for (self.gpus, samples) |gpu, *sample| {
        const fields = &sample.rocm;
        const handle = gpu.handle;

        var metrics: c.amdsmi_gpu_metrics_t = undefined;
        if (api.amdsmi_get_gpu_metrics_info.call(.{ handle, &metrics })) |_| {
            if (known(metrics.current_socket_power) orelse known(metrics.average_socket_power)) |watts| fields.power_mw = watts * 1000;
        } else |_| {}
        var power: c.amdsmi_power_info_t = undefined;
        if (api.amdsmi_get_power_info.call(.{ handle, &power })) |_| {
            if (known(power.power_limit)) |microwatts| fields.power_limit_mw = microwatts / 1000;
        } else |_| {}
        var celsius: i64 = 0;
        if (api.amdsmi_get_temp_metric.call(.{ handle, c.AMDSMI_TEMPERATURE_TYPE_EDGE, c.AMDSMI_TEMP_CURRENT, &celsius })) |_| {
            fields.temperature = std.math.cast(u64, celsius);
        } else |_| {}
        var speed: i64 = 0;
        if (api.amdsmi_get_gpu_fan_speed.call(.{ handle, 0, &speed })) |_| {
            if (std.math.cast(u64, speed)) |pwm| fields.fan_speed_percent = smi.hwmon.pwmPercent(pwm);
        } else |_| {}
        var activity: c.amdsmi_engine_usage_t = undefined;
        if (api.amdsmi_get_gpu_activity.call(.{ handle, &activity })) |_| {
            // Not a percentage when unsupported: 0xFFFF.
            if (activity.gfx_activity <= 100) fields.util_percent = activity.gfx_activity;
        } else |_| {}

        var clock: c.amdsmi_clk_info_t = undefined;
        if (api.amdsmi_get_clock_info.call(.{ handle, c.AMDSMI_CLK_TYPE_SYS, &clock })) |_| {
            fields.clock_graphics_mhz = known(clock.clk);
            fields.clock_graphics_max_mhz = known(clock.max_clk);
        } else |_| {}
        if (api.amdsmi_get_clock_info.call(.{ handle, c.AMDSMI_CLK_TYPE_SOC, &clock })) |_| {
            fields.clock_soc_mhz = known(clock.clk);
        } else |_| {}
        if (api.amdsmi_get_clock_info.call(.{ handle, c.AMDSMI_CLK_TYPE_MEM, &clock })) |_| {
            fields.clock_mem_mhz = known(clock.clk);
            fields.clock_mem_max_mhz = known(clock.max_clk);
        } else |_| {}

        var bytes: u64 = 0;
        fields.mem_used_bytes = if (api.amdsmi_get_gpu_memory_usage.call(.{ handle, c.AMDSMI_MEM_TYPE_VRAM, &bytes })) |_| bytes else |_| null;
        fields.mem_total_bytes = if (api.amdsmi_get_gpu_memory_total.call(.{ handle, c.AMDSMI_MEM_TYPE_VRAM, &bytes })) |_| bytes else |_| null;

        var pcie: c.amdsmi_pcie_info_t = undefined;
        if (api.amdsmi_get_pcie_info.call(.{ handle, &pcie })) |_| {
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
    self.curr_busy.clearRetainingCapacity();
    defer std.mem.swap(@TypeOf(self.prev_busy), &self.prev_busy, &self.curr_busy);

    const now: std.Io.Timestamp = .now(self.io, .awake);
    const elapsed: ?std.Io.Duration = if (self.last_poll) |last|
        if (now.nanoseconds > last.nanoseconds) last.durationTo(now) else null
    else
        null;
    self.last_poll = now;
    var read_any = self.gpus.len == 0;

    // amdsmi reports the VRAM of a process over all GPUs, fdinfo per GPU.
    var vram: std.AutoHashMapUnmanaged(struct { u32, u16 }, u64) = .empty;
    var clients: smi.fdinfo.Clients = .init(self.io, scratch);
    defer clients.deinit();
    while (clients.next()) |client| {
        const device = self.deviceOf(client.info.pdev orelse continue) orelse continue;
        const bytes = try vram.getOrPutValue(scratch, .{ client.pid, device }, 0);
        for (client.info.regions()) |region| {
            if (std.mem.eql(u8, region.name, "vram")) bytes.value_ptr.* += region.bytes;
        }
    }

    for (self.gpus, 0..) |gpu, index| {
        const device: u16 = @intCast(index);
        const list = self.processList(scratch, gpu.handle) catch continue;
        read_any = true;
        for (list) |process| {
            const key: BusyKey = .{ .pid = process.pid, .device = device };
            const busy: std.Io.Duration = .fromNanoseconds(process.engine_usage.gfx +| process.engine_usage.enc);
            try self.curr_busy.put(self.gpa, key, busy);
            const util: ?u16 = if (self.prev_busy.get(key)) |prev|
                if (elapsed != null and busy.nanoseconds >= prev.nanoseconds)
                    @intCast(@min(@divTrunc((busy.nanoseconds - prev.nanoseconds) * 100, elapsed.?.nanoseconds), 100))
                else
                    null
            else
                null;
            try out.append(.{
                .pid = process.pid,
                .device_idx = device,
                .dev_util_percent = util,
                .dev_mem_kib = if (vram.get(.{ process.pid, device })) |bytes| bytes / 1024 else null,
            });
        }
    }
    if (!read_any) return error.Unavailable;
}

fn deinit(platform: *smi.Platform) void {
    const self: *AmdSmi = @alignCast(@fieldParentPtr("platform", platform));
    self.prev_busy.deinit(self.gpa);
    self.curr_busy.deinit(self.gpa);
}

const BusyKey = struct { pid: u32, device: u16 };

const Gpu = struct {
    handle: c.amdsmi_processor_handle,
    address: ?[12]u8,
};

fn processList(self: *AmdSmi, scratch: std.mem.Allocator, handle: c.amdsmi_processor_handle) ![]const c.amdsmi_proc_info_t {
    var count: u32 = 0;
    try Api.check(self.listQuietly(handle, &count, null));
    if (count == 0) return &.{};

    const list = try scratch.alloc(c.amdsmi_proc_info_t, count);
    const status = self.listQuietly(handle, &count, list.ptr);
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

fn deviceOf(self: *const AmdSmi, pdev: []const u8) ?u16 {
    for (self.gpus, 0..) |gpu, index| {
        const address = gpu.address orelse continue;
        if (std.mem.eql(u8, &address, pdev)) return @intCast(index);
    }
    return null;
}

/// amdsmi reports a value it cannot read as the largest of its type.
fn known(value: anytype) ?u64 {
    return if (value == std.math.maxInt(@TypeOf(value))) null else value;
}

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
