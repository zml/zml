const std = @import("std");
const c = @import("c");
const smi = @import("zml-smi");

const AmdSmi = @This();

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
    amdsmi_init: *const @TypeOf(c.amdsmi_init),
    amdsmi_shut_down: *const @TypeOf(c.amdsmi_shut_down),
    amdsmi_get_socket_handles: *const @TypeOf(c.amdsmi_get_socket_handles),
    amdsmi_get_processor_handles: *const @TypeOf(c.amdsmi_get_processor_handles),
    amdsmi_get_gpu_asic_info: *const @TypeOf(c.amdsmi_get_gpu_asic_info),
    amdsmi_get_gpu_driver_info: *const @TypeOf(c.amdsmi_get_gpu_driver_info),
    amdsmi_get_gpu_bdf_id: *const @TypeOf(c.amdsmi_get_gpu_bdf_id),
    amdsmi_get_gpu_metrics_info: *const @TypeOf(c.amdsmi_get_gpu_metrics_info),
    amdsmi_get_power_info: *const @TypeOf(c.amdsmi_get_power_info),
    amdsmi_get_temp_metric: *const @TypeOf(c.amdsmi_get_temp_metric),
    amdsmi_get_gpu_fan_speed: *const @TypeOf(c.amdsmi_get_gpu_fan_speed),
    amdsmi_get_gpu_activity: *const @TypeOf(c.amdsmi_get_gpu_activity),
    amdsmi_get_clock_info: *const @TypeOf(c.amdsmi_get_clock_info),
    amdsmi_get_gpu_memory_total: *const @TypeOf(c.amdsmi_get_gpu_memory_total),
    amdsmi_get_gpu_memory_usage: *const @TypeOf(c.amdsmi_get_gpu_memory_usage),
    amdsmi_get_pcie_info: *const @TypeOf(c.amdsmi_get_pcie_info),
    amdsmi_get_gpu_process_list: *const @TypeOf(c.amdsmi_get_gpu_process_list),
};

pub fn detect(io: std.Io) bool {
    return smi.sysfs.exists(io, "/dev/kfd");
}

pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !AmdSmi {
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox = smi.sandbox.path(gpa, io, &buffer) orelse return error.LibraryNotFound;

    var path: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const api = try smi.dynlib.open(Api, &.{try std.fmt.bufPrintSentinel(&path, "{s}/lib/libamd_smi.so.27", .{sandbox}, 0)});
    try check(api.amdsmi_init(c.AMDSMI_INIT_AMD_GPUS));
    errdefer _ = api.amdsmi_shut_down();

    var socket_count: u32 = 0;
    try check(api.amdsmi_get_socket_handles(&socket_count, null));
    const sockets = try gpa.alloc(c.amdsmi_socket_handle, socket_count);
    defer gpa.free(sockets);
    try check(api.amdsmi_get_socket_handles(&socket_count, sockets.ptr));

    var gpus: std.ArrayList(Gpu) = .empty;
    var devices: std.ArrayList(smi.Device) = .empty;
    for (sockets[0..socket_count]) |socket| {
        var count: u32 = 0;
        try check(api.amdsmi_get_processor_handles(socket, &count, null));
        const handles = try gpa.alloc(c.amdsmi_processor_handle, count);
        defer gpa.free(handles);
        try check(api.amdsmi_get_processor_handles(socket, &count, handles.ptr));

        for (handles[0..count]) |handle| {
            var bdf: u64 = 0;
            var asic: c.amdsmi_asic_info_t = undefined;
            var driver: c.amdsmi_driver_info_t = undefined;
            try gpus.append(arena, .{ .handle = handle, .address = if (ok(api.amdsmi_get_gpu_bdf_id(handle, &bdf))) pciAddress(bdf) else null });

            // amdsmi reports the version of an in-tree driver, which has none, as N/A.
            const version = if (ok(api.amdsmi_get_gpu_driver_info(handle, &driver))) std.mem.sliceTo(&driver.driver_version, 0) else "N/A";
            try devices.append(arena, .{
                .target = .rocm,
                .name = if (ok(api.amdsmi_get_gpu_asic_info(handle, &asic))) try arena.dupe(u8, std.mem.sliceTo(&asic.market_name, 0)) else null,
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
        if (ok(api.amdsmi_get_gpu_metrics_info(handle, &metrics))) {
            if (known(metrics.current_socket_power) orelse known(metrics.average_socket_power)) |watts| fields.power_mw = watts * 1000;
        }

        var power: c.amdsmi_power_info_t = undefined;
        if (ok(api.amdsmi_get_power_info(handle, &power))) {
            if (known(power.power_limit)) |microwatts| fields.power_limit_mw = microwatts / 1000;
        }

        var celsius: i64 = 0;
        if (ok(api.amdsmi_get_temp_metric(handle, c.AMDSMI_TEMPERATURE_TYPE_EDGE, c.AMDSMI_TEMP_CURRENT, &celsius))) {
            fields.temperature = std.math.cast(u64, celsius);
        }

        var speed: i64 = 0;
        if (ok(api.amdsmi_get_gpu_fan_speed(handle, 0, &speed))) {
            if (std.math.cast(u64, speed)) |pwm| fields.fan_speed_percent = smi.hwmon.pwmPercent(pwm);
        }

        var activity: c.amdsmi_engine_usage_t = undefined;
        if (ok(api.amdsmi_get_gpu_activity(handle, &activity))) {
            // Not a percentage when unsupported: 0xFFFF.
            if (activity.gfx_activity <= 100) fields.util_percent = activity.gfx_activity;
        }

        var clock: c.amdsmi_clk_info_t = undefined;
        if (ok(api.amdsmi_get_clock_info(handle, c.AMDSMI_CLK_TYPE_SYS, &clock))) {
            fields.clock_graphics_mhz = known(clock.clk);
            fields.clock_graphics_max_mhz = known(clock.max_clk);
        }

        if (ok(api.amdsmi_get_clock_info(handle, c.AMDSMI_CLK_TYPE_SOC, &clock))) {
            fields.clock_soc_mhz = known(clock.clk);
        }

        if (ok(api.amdsmi_get_clock_info(handle, c.AMDSMI_CLK_TYPE_MEM, &clock))) {
            fields.clock_mem_mhz = known(clock.clk);
            fields.clock_mem_max_mhz = known(clock.max_clk);
        }

        var bytes: u64 = 0;
        fields.mem_used_bytes = if (ok(api.amdsmi_get_gpu_memory_usage(handle, c.AMDSMI_MEM_TYPE_VRAM, &bytes))) known(bytes) else null;
        fields.mem_total_bytes = if (ok(api.amdsmi_get_gpu_memory_total(handle, c.AMDSMI_MEM_TYPE_VRAM, &bytes))) known(bytes) else null;

        var pcie: c.amdsmi_pcie_info_t = undefined;
        if (ok(api.amdsmi_get_pcie_info(handle, &pcie))) {
            const link: smi.pci.Link = .{ .gen = known(pcie.pcie_static.pcie_interface_version), .width = known(pcie.pcie_metric.pcie_width) };
            fields.pcie_link_gen = link.gen;
            fields.pcie_link_width = link.width;
            // The rate comes from the generation
            fields.pcie_bandwidth_mbps = link.mbps();
        }
    }
}

fn processes(platform: *smi.Platform, scratch: std.mem.Allocator, _: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
    const self: *AmdSmi = @alignCast(@fieldParentPtr("platform", platform));

    self.curr_busy.clearRetainingCapacity();
    defer std.mem.swap(@TypeOf(self.prev_busy), &self.prev_busy, &self.curr_busy);

    const now: std.Io.Timestamp = .now(self.io, .awake);
    const elapsed: ?std.Io.Duration = blk: {
        const last = self.last_poll orelse break :blk null;
        if (now.nanoseconds <= last.nanoseconds) break :blk null;

        break :blk last.durationTo(now);
    };

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

            const util: ?u16 = blk: {
                const dt = elapsed orelse break :blk null;
                const prev = self.prev_busy.get(key) orelse break :blk null;
                if (busy.nanoseconds < prev.nanoseconds) break :blk null;

                break :blk @intCast(@min(@divTrunc((busy.nanoseconds - prev.nanoseconds) * 100, dt.nanoseconds), 100));
            };

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

    _ = self.api.amdsmi_shut_down();
    self.prev_busy.deinit(self.gpa);
    self.curr_busy.deinit(self.gpa);
}

const BusyKey = struct { pid: u32, device: u16 };

const Gpu = struct {
    handle: c.amdsmi_processor_handle,
    address: ?[12]u8,
};

/// amdsmi prints while it lists processes, so stderr points at the null device under the stderr lock.
fn processList(self: *AmdSmi, scratch: std.mem.Allocator, handle: c.amdsmi_processor_handle) ![]const c.amdsmi_proc_info_t {
    _ = std.debug.lockStderr(&.{});
    defer std.debug.unlockStderr();

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
    try check(self.api.amdsmi_get_gpu_process_list(handle, &count, null));
    if (count == 0) return &.{};

    const list = try scratch.alloc(c.amdsmi_proc_info_t, count);
    const status = self.api.amdsmi_get_gpu_process_list(handle, &count, list.ptr);
    if (status != c.AMDSMI_STATUS_OUT_OF_RESOURCES) try check(status);

    return list[0..@min(count, list.len)];
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

fn ok(status: c.amdsmi_status_t) bool {
    return status == c.AMDSMI_STATUS_SUCCESS;
}

fn check(status: c.amdsmi_status_t) !void {
    return switch (status) {
        c.AMDSMI_STATUS_SUCCESS => {},
        c.AMDSMI_STATUS_INVAL => error.InvalidArgument,
        c.AMDSMI_STATUS_NOT_SUPPORTED => error.NotSupported,
        c.AMDSMI_STATUS_NOT_YET_IMPLEMENTED => error.NotYetImplemented,
        c.AMDSMI_STATUS_FAIL_LOAD_MODULE => error.FailLoadModule,
        c.AMDSMI_STATUS_FAIL_LOAD_SYMBOL => error.FailLoadSymbol,
        c.AMDSMI_STATUS_DRM_ERROR => error.DrmError,
        c.AMDSMI_STATUS_API_FAILED => error.ApiFailed,
        c.AMDSMI_STATUS_TIMEOUT => error.Timeout,
        c.AMDSMI_STATUS_RETRY => error.Retry,
        c.AMDSMI_STATUS_NO_PERM => error.NoPermission,
        c.AMDSMI_STATUS_INTERRUPT => error.Interrupt,
        c.AMDSMI_STATUS_IO => error.IoError,
        c.AMDSMI_STATUS_ADDRESS_FAULT => error.AddressFault,
        c.AMDSMI_STATUS_FILE_ERROR => error.FileError,
        c.AMDSMI_STATUS_OUT_OF_RESOURCES => error.OutOfResources,
        c.AMDSMI_STATUS_INTERNAL_EXCEPTION => error.InternalException,
        c.AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS => error.InputOutOfBounds,
        c.AMDSMI_STATUS_INIT_ERROR => error.InitError,
        c.AMDSMI_STATUS_REFCOUNT_OVERFLOW => error.RefcountOverflow,
        c.AMDSMI_STATUS_DIRECTORY_NOT_FOUND => error.DirectoryNotFound,
        c.AMDSMI_STATUS_BUSY => error.Busy,
        c.AMDSMI_STATUS_NOT_FOUND => error.NotFound,
        c.AMDSMI_STATUS_NOT_INIT => error.NotInit,
        c.AMDSMI_STATUS_NO_SLOT => error.NoSlot,
        c.AMDSMI_STATUS_DRIVER_NOT_LOADED => error.DriverNotLoaded,
        c.AMDSMI_STATUS_NO_DATA => error.NoData,
        c.AMDSMI_STATUS_INSUFFICIENT_SIZE => error.InsufficientSize,
        c.AMDSMI_STATUS_UNEXPECTED_SIZE => error.UnexpectedSize,
        c.AMDSMI_STATUS_UNEXPECTED_DATA => error.UnexpectedData,
        else => error.Unknown,
    };
}

test pciAddress {
    try std.testing.expectEqualStrings("0000:03:00.0", &pciAddress(0x0300).?);
    // Bits 28 to 31 hold a partition id, which is not part of the address.
    try std.testing.expectEqualStrings("0001:c1:00.1", &pciAddress(0x1_1000_c101).?);
}
