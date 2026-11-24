const std = @import("std");
const smi = @import("zml-smi");

const linux = std.os.linux;

const OneApi = @This();

platform: smi.Platform,
io: std.Io,
gpa: std.mem.Allocator,
gpus: []Gpu,
prev_usage: std.AutoHashMapUnmanaged(ProcessKey, Reading) = .empty,
curr_usage: std.AutoHashMapUnmanaged(ProcessKey, Reading) = .empty,
last_scan: ?std.Io.Timestamp = null,
rows: ?[]const smi.ProcessInfo = null,

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes, .deinit = deinit };

pub fn detect(io: std.Io) bool {
    const drm = smi.sysfs.Dir.open(io, "/sys/class/drm") catch return false;
    defer drm.close();

    var it = drm.iterate();
    while (it.next(io) catch null) |entry| {
        const node = renderNode(entry.name) orelse continue;
        const dir = openDevice(drm, node) catch continue;
        defer dir.close();
        if ((dir.int(u16, "vendor") catch 0) == intel) return true;
    }

    return false;
}

pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !OneApi {
    const drm = try smi.sysfs.Dir.open(io, "/sys/class/drm");
    defer drm.close();

    var nodes: std.ArrayList(u32) = .empty;
    var it = drm.iterate();
    while (try it.next(io)) |entry| {
        if (renderNode(entry.name)) |node| try nodes.append(arena, node);
    }
    std.mem.sort(u32, nodes.items, {}, std.sort.asc(u32));

    const driver_version = driverVersion(arena, io);
    var gpus: std.ArrayList(Gpu) = .empty;
    errdefer for (gpus.items) |gpu| gpu.close(io);

    var devices: std.ArrayList(smi.Device) = .empty;
    for (nodes.items) |node| {
        const dir = openDevice(drm, node) catch continue;
        const gpu = openGpu(arena, io, dir, node) catch {
            dir.close();
            continue;
        };
        gpus.append(arena, gpu) catch |err| {
            gpu.close(io);
            return err;
        };
        try devices.append(arena, .{ .target = .oneapi, .name = try name(arena, io, gpu), .driver_version = driver_version });
    }

    return .{
        .platform = .{ .devices = devices.items, .two_readings = true, .vtable = &vtable },
        .io = io,
        .gpa = gpa,
        .gpus = gpus.items,
    };
}

fn poll(platform: *smi.Platform, scratch: std.mem.Allocator, samples: []smi.Sample) void {
    const self: *OneApi = @alignCast(@fieldParentPtr("platform", platform));

    for (self.gpus, samples) |*gpu, *sample| {
        const fields = &sample.oneapi;

        fields.clock_graphics_mhz = gpu.dir.int(u64, "tile0/gt0/freq0/act_freq") catch gpu.dir.int(u64, "tile0/gt0/freq0/cur_freq") catch null;
        fields.clock_graphics_max_mhz = gpu.dir.int(u64, "tile0/gt0/freq0/max_freq") catch null;
        fields.mem_total_bytes = gpu.mem_total_bytes;

        if (gpu.render) |render| {
            if (gpu.driver == .xe) {
                if (vram(render, .xe)) |memory| fields.mem_used_bytes = memory.used;
            }
        }

        const link = smi.pci.link(gpu.link, .current);
        fields.pcie_link_gen = link.gen;
        fields.pcie_link_width = link.width;
        fields.pcie_bandwidth_mbps = link.mbps();

        const hwmon = gpu.hwmon orelse continue;

        if (hwmon.temperature) |file| {
            if (hwmon.dir.int(u64, file) catch null) |millidegrees| fields.temperature = millidegrees / 1000;
        }

        if (hwmon.power_limit) |file| {
            if (hwmon.dir.int(u64, file) catch null) |microwatts| fields.power_limit_mw = microwatts / 1000;
        }

        if (hwmon.energy) |file| {
            if (hwmon.dir.int(u64, file)) |microjoules| {
                const now: std.Io.Timestamp = .now(self.io, .awake);
                if (gpu.last_energy) |prev| {
                    if (microjoules >= prev.microjoules and now.nanoseconds > prev.at.nanoseconds) {
                        const dt = prev.at.durationTo(now);
                        fields.power_mw = std.math.lossyCast(u64, @as(u128, microjoules - prev.microjoules) * std.time.ns_per_ms / @as(u128, @intCast(dt.nanoseconds)));
                    }
                }
                gpu.last_energy = .{ .microjoules = microjoules, .at = now };
            } else |_| {}
        }

        if (hwmon.fan) |fan| fields.fan_speed_percent = fan.percent(hwmon.dir);
    }

    self.scan(scratch, samples);
}

fn processes(platform: *smi.Platform, _: std.mem.Allocator, _: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
    const self: *OneApi = @alignCast(@fieldParentPtr("platform", platform));

    for (self.rows orelse return error.Unavailable) |row| {
        try out.append(row);
    }
}

fn deinit(platform: *smi.Platform) void {
    const self: *OneApi = @alignCast(@fieldParentPtr("platform", platform));

    for (self.gpus) |gpu| gpu.close(self.io);

    self.prev_usage.deinit(self.gpa);
    self.curr_usage.deinit(self.gpa);
}

const Gpu = struct {
    dir: smi.sysfs.Dir,
    link: smi.sysfs.Dir,
    address: [12]u8,
    hwmon: ?Hwmon,
    last_energy: ?struct { microjoules: u64, at: std.Io.Timestamp } = null,
    driver: Driver,
    render: ?std.Io.File,
    mem_total_bytes: ?u64,

    fn close(gpu: Gpu, io: std.Io) void {
        gpu.dir.close();
        gpu.link.close();
        if (gpu.hwmon) |hwmon| hwmon.dir.close();
        if (gpu.render) |render| render.close(io);
    }
};

const Driver = enum { xe, i915, other };

const Hwmon = struct {
    dir: smi.sysfs.Dir,
    temperature: ?[]const u8,
    energy: ?[]const u8,
    power_limit: ?[]const u8,
    fan: ?Fan,
};

const Fan = union(enum) {
    pwm: []const u8,
    speed: struct { input: []const u8, max: []const u8 },

    fn find(dir: smi.sysfs.Dir) ?Fan {
        const pwm = comptime smi.hwmon.numbered("pwm{d}");
        if (smi.hwmon.firstReadable(dir, &pwm)) |path| return .{ .pwm = path };

        const inputs = comptime smi.hwmon.numbered("fan{d}_input");
        const limits = comptime smi.hwmon.numbered("fan{d}_max");

        for (inputs, limits) |input, max| {
            _ = dir.int(u64, input) catch continue;
            if ((dir.int(u64, max) catch 0) != 0) return .{ .speed = .{ .input = input, .max = max } };
        }

        return null;
    }

    fn percent(fan: Fan, dir: smi.sysfs.Dir) ?u64 {
        switch (fan) {
            .pwm => |file| return smi.hwmon.pwmPercent(dir.int(u64, file) catch return null),
            .speed => |files| {
                const input = dir.int(u64, files.input) catch return null;
                const max = dir.int(u64, files.max) catch return null;
                return if (max > 0) @min(input * 100 / max, 100) else null;
            },
        }
    }
};

const ProcessKey = struct { pid: u32, device: u16 };

const intel = 0x8086;

fn renderNode(entry: []const u8) ?u32 {
    const number = std.mem.cutPrefix(u8, entry, "renderD") orelse return null;
    return std.fmt.parseInt(u32, number, 10) catch null;
}

fn openDevice(drm: smi.sysfs.Dir, node: u32) !smi.sysfs.Dir {
    var path: [32]u8 = undefined;
    return drm.openDir(try std.fmt.bufPrint(&path, "renderD{d}/device", .{node}));
}

fn openGpu(arena: std.mem.Allocator, io: std.Io, dir: smi.sysfs.Dir, node: u32) !Gpu {
    if (try dir.int(u16, "vendor") != intel) return error.NotIntel;

    var buffer: [2048]u8 = undefined;
    const slot = try dir.field("uevent", "PCI_SLOT_NAME", &buffer);
    var address: [12]u8 = undefined;
    if (slot.len != address.len) return error.NotFound;
    @memcpy(&address, slot);

    // The GPU sits behind the card's own switch, whose inner links report 2.5 GT/s x1. The slot's link is on the switch's upstream port.
    const upstream = try dir.openDir("../..");
    var link: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const driver_name: ?[]const u8 = if (dir.readLink("driver", &link)) |path| std.fs.path.basename(path) else |_| null;
    const driver = std.meta.stringToEnum(Driver, driver_name orelse "") orelse .other;
    var path: [32]u8 = undefined;
    const render = std.Io.Dir.cwd().openFile(io, std.fmt.bufPrint(&path, "/dev/dri/renderD{d}", .{node}) catch unreachable, .{ .mode = .read_write }) catch null;

    const mem_total_bytes: ?u64 = blk: {
        const file = render orelse break :blk null;
        const memory = vram(file, driver) orelse break :blk null;

        break :blk memory.total;
    };

    return .{
        .dir = dir,
        .link = upstream,
        .address = address,
        .hwmon = openHwmon(arena, dir, driver_name),
        .driver = driver,
        .render = render,
        .mem_total_bytes = mem_total_bytes,
    };
}

// https://github.com/torvalds/linux/blob/c03c1a14507dfaf8b856c2608f9ac216021fc222/include/uapi/drm/xe_drm.h
const xe_drm = struct {
    const DeviceQuery = extern struct { extensions: u64 = 0, query: u32, size: u32 = 0, data: u64 = 0, reserved: [2]u64 = @splat(0) };
    const MemRegion = extern struct { mem_class: u16, instance: u16, min_page_size: u32, total_size: u64, used: u64, cpu_visible_size: u64, cpu_visible_used: u64, reserved: [6]u64 };
    const MemRegions = extern struct { num_mem_regions: u32, pad: u32, mem_regions: [16]MemRegion };
    const device_query = linux.IOCTL.IOWR('d', 0x40, DeviceQuery);
    const query_mem_regions = 1;
    const class_vram = 1;
};

// https://github.com/torvalds/linux/blob/c03c1a14507dfaf8b856c2608f9ac216021fc222/include/uapi/drm/i915_drm.h
const i915_drm = struct {
    const QueryItem = extern struct { query_id: u64, length: i32 = 0, flags: u32 = 0, data_ptr: u64 = 0 };
    const Query = extern struct { num_items: u32, flags: u32 = 0, items_ptr: u64 };
    const MemoryRegionInfo = extern struct { memory_class: u16, memory_instance: u16, rsvd0: u32, probed_size: u64, unallocated_size: u64, rsvd1: [8]u64 };
    const MemoryRegions = extern struct { num_regions: u32, rsvd: [3]u32, regions: [16]MemoryRegionInfo };
    const query = linux.IOCTL.IOWR('d', 0x40 + 0x39, Query);
    const query_memory_regions = 4;
    const class_device = 1;
};

const Vram = struct { total: u64, used: ?u64 };

/// The VRAM regions of the driver's memory region query. i915 reports the used bytes only to CAP_PERFMON.
fn vram(render: std.Io.File, driver: Driver) ?Vram {
    var total: u64 = 0;
    var used: ?u64 = null;
    switch (driver) {
        .xe => {
            var reply: xe_drm.MemRegions = undefined;
            var query: xe_drm.DeviceQuery = .{ .query = xe_drm.query_mem_regions };
            if (linux.errno(linux.ioctl(render.handle, xe_drm.device_query, @intFromPtr(&query))) != .SUCCESS or query.size > @sizeOf(xe_drm.MemRegions)) return null;
            query.data = @intFromPtr(&reply);
            if (linux.errno(linux.ioctl(render.handle, xe_drm.device_query, @intFromPtr(&query))) != .SUCCESS) return null;
            var bytes: u64 = 0;
            for (reply.mem_regions[0..@min(reply.num_mem_regions, reply.mem_regions.len)]) |region| {
                if (region.mem_class != xe_drm.class_vram) continue;
                total += region.total_size;
                bytes += region.used;
            }
            used = bytes;
        },
        .i915 => {
            var reply: i915_drm.MemoryRegions = undefined;
            var item: i915_drm.QueryItem = .{ .query_id = i915_drm.query_memory_regions };
            var query: i915_drm.Query = .{ .num_items = 1, .items_ptr = @intFromPtr(&item) };
            if (linux.errno(linux.ioctl(render.handle, i915_drm.query, @intFromPtr(&query))) != .SUCCESS or item.length <= 0 or item.length > @sizeOf(i915_drm.MemoryRegions)) return null;
            item.data_ptr = @intFromPtr(&reply);
            if (linux.errno(linux.ioctl(render.handle, i915_drm.query, @intFromPtr(&query))) != .SUCCESS or item.length <= 0) return null;
            for (reply.regions[0..@min(reply.num_regions, reply.regions.len)]) |region| {
                if (region.memory_class == i915_drm.class_device) total += region.probed_size;
            }
        },
        .other => {},
    }
    return if (total > 0) .{ .total = total, .used = used } else null;
}

fn openHwmon(arena: std.mem.Allocator, device: smi.sysfs.Dir, driver: ?[]const u8) ?Hwmon {
    const dir = smi.hwmon.find(device, driver) catch return null;
    return .{
        .dir = dir,
        .temperature = smi.hwmon.labeled(arena, dir, "temp", "input", "pkg") orelse
            smi.hwmon.labeled(arena, dir, "temp", "input", "gpu") orelse
            smi.hwmon.firstReadable(dir, &.{ "temp2_input", "temp1_input" }),
        .energy = smi.hwmon.labeled(arena, dir, "energy", "input", "card") orelse
            smi.hwmon.labeled(arena, dir, "energy", "input", "pkg") orelse
            smi.hwmon.firstReadable(dir, &.{"energy1_input"}),
        .power_limit = smi.hwmon.firstReadable(dir, &.{ "power1_cap", "power1_max" }),
        .fan = .find(dir),
    };
}

const Reading = struct {
    cycles: ?u64 = null,
    total_cycles: ?u64 = null,
    busy: ?std.Io.Duration = null,
    encoder: ?std.Io.Duration = null,
    decoder: ?std.Io.Duration = null,
};

const Usage = struct {
    row: smi.ProcessInfo,
    reading: Reading = .{},
};

const DeviceUsage = struct {
    cycles: u64 = 0,
    total_cycles: u64 = 0,
    busy: std.Io.Duration = .zero,
    encoder: std.Io.Duration = .zero,
    decoder: std.Io.Duration = .zero,
    has_cycles: bool = false,
    has_encoder: bool = false,
    has_decoder: bool = false,
};

fn scan(self: *OneApi, scratch: std.mem.Allocator, samples: []smi.Sample) void {
    self.curr_usage.clearRetainingCapacity();
    defer std.mem.swap(@TypeOf(self.prev_usage), &self.prev_usage, &self.curr_usage);

    const now: std.Io.Timestamp = .now(self.io, .awake);
    const elapsed: ?std.Io.Duration = blk: {
        const last = self.last_scan orelse break :blk null;
        if (now.nanoseconds <= last.nanoseconds) break :blk null;

        break :blk last.durationTo(now);
    };

    self.last_scan = now;
    self.rows = null;

    var usages: std.AutoArrayHashMapUnmanaged(ProcessKey, Usage) = .empty;
    var clients: smi.fdinfo.Clients = .init(self.io, scratch);
    defer clients.deinit();
    while (clients.next()) |client| {
        const device = self.deviceOf(client.info.pdev orelse continue) orelse continue;
        const usage = (usages.getOrPutValue(scratch, .{ .pid = client.pid, .device = device }, .{ .row = .{ .pid = client.pid, .device_idx = device } }) catch continue).value_ptr;
        if (memoryBytes(&client.info)) |bytes| usage.row.dev_mem_kib = (usage.row.dev_mem_kib orelse 0) + bytes / 1024;
        for (client.info.engines()) |engine| {
            if (engine.total_cycles) |total| {
                usage.reading.cycles = (usage.reading.cycles orelse 0) + (engine.cycles orelse 0);
                usage.reading.total_cycles = @max(usage.reading.total_cycles orelse 0, total);
            } else if (engine.busy) |busy| {
                addDuration(&usage.reading.busy, busy);
                if (std.ascii.findIgnoreCase(engine.class, "encode") != null) {
                    addDuration(&usage.reading.encoder, busy);
                } else if (std.ascii.findIgnoreCase(engine.class, "decode") != null) {
                    addDuration(&usage.reading.decoder, busy);
                }
            }
        }
    }

    const devices = scratch.alloc(DeviceUsage, self.gpus.len) catch return;
    @memset(devices, .{});
    const rows = scratch.alloc(smi.ProcessInfo, usages.count()) catch return;
    for (rows, usages.keys(), usages.values()) |*row, key, usage| {
        const device = &devices[key.device];
        row.* = usage.row;
        device.has_encoder = device.has_encoder or usage.reading.encoder != null;
        device.has_decoder = device.has_decoder or usage.reading.decoder != null;

        self.curr_usage.put(self.gpa, key, usage.reading) catch {};
        const prev = self.prev_usage.get(key) orelse continue;

        if (usage.reading.total_cycles) |total| {
            const prev_total = prev.total_cycles orelse continue;
            const cycles = usage.reading.cycles orelse 0;
            const prev_cycles = prev.cycles orelse continue;
            if (total >= prev_total and cycles >= prev_cycles) {
                const busy = cycles - prev_cycles;
                const dt_cycles = total - prev_total;
                row.dev_util_percent = cyclePercent(busy, dt_cycles);
                device.has_cycles = true;
                device.cycles +|= busy;
                device.total_cycles = @max(device.total_cycles, dt_cycles);
            }
        } else if (durationDelta(usage.reading.busy, prev.busy)) |busy| {
            if (elapsed) |dt| row.dev_util_percent = timePercent(busy, dt);
            device.busy.nanoseconds += busy.nanoseconds;
        }
        if (durationDelta(usage.reading.encoder, prev.encoder)) |busy| {
            device.encoder.nanoseconds += busy.nanoseconds;
        }
        if (durationDelta(usage.reading.decoder, prev.decoder)) |busy| {
            device.decoder.nanoseconds += busy.nanoseconds;
        }
    }
    self.rows = rows;

    for (samples, devices, 0..) |*sample, device, index| {
        const fields = &sample.oneapi;
        fields.util_percent = blk: {
            if (device.has_cycles) break :blk cyclePercent(device.cycles, device.total_cycles);
            if (elapsed) |dt| break :blk timePercent(device.busy, dt);

            break :blk 0;
        };

        if (device.has_encoder) {
            fields.encoder_util_percent = if (elapsed) |dt| timePercent(device.encoder, dt) else 0;
        }

        if (device.has_decoder) {
            fields.decoder_util_percent = if (elapsed) |dt| timePercent(device.decoder, dt) else 0;
        }

        // Without the query's count, the clients we can read: those of other users are missing.
        if (fields.mem_used_bytes != null) continue;
        var mem_kib: u64 = 0;
        for (rows) |row| {
            if (row.device_idx == index) mem_kib += row.dev_mem_kib orelse 0;
        }

        fields.mem_used_bytes = mem_kib * 1024;
    }
}

fn durationDelta(curr: ?std.Io.Duration, prev: ?std.Io.Duration) ?std.Io.Duration {
    const c = curr orelse return null;
    const p = prev orelse return null;

    return if (c.nanoseconds >= p.nanoseconds) .fromNanoseconds(c.nanoseconds - p.nanoseconds) else null;
}

fn timePercent(busy: std.Io.Duration, elapsed: std.Io.Duration) u8 {
    return @intCast(@min(@divTrunc(busy.nanoseconds * 100, elapsed.nanoseconds), 100));
}

fn cyclePercent(busy: u64, total: u64) u8 {
    if (total == 0) return 0;
    return @intCast(@min(@as(u128, busy) * 100 / total, 100));
}

fn addDuration(sum: *?std.Io.Duration, value: std.Io.Duration) void {
    if (sum.*) |*total| total.nanoseconds += value.nanoseconds else sum.* = value;
}

fn deviceOf(self: *const OneApi, pdev: []const u8) ?u16 {
    for (self.gpus, 0..) |gpu, index| {
        if (std.mem.eql(u8, &gpu.address, pdev)) return @intCast(index);
    }
    return null;
}

/// Resident device memory (xe's vram, i915's local), or resident system memory on a GPU without its own.
fn memoryBytes(info: *const smi.fdinfo.Fdinfo) ?u64 {
    var device: ?u64 = null;
    var system: ?u64 = null;

    for (info.regions()) |region| {
        if (std.mem.startsWith(u8, region.name, "vram") or std.mem.startsWith(u8, region.name, "local")) {
            device = (device orelse 0) + region.bytes;
        } else {
            system = (system orelse 0) + region.bytes;
        }
    }

    return device orelse system;
}

fn driverVersion(arena: std.mem.Allocator, io: std.Io) ?[]const u8 {
    const module = smi.sysfs.Dir.open(io, "/sys/module") catch return null;
    defer module.close();

    var buffer: [128]u8 = undefined;
    for ([_][]const u8{ "xe/version", "i915/version" }) |file| {
        const version = module.read(file, &buffer) catch continue;
        if (version.len > 0) return arena.dupe(u8, version) catch null;
    }

    return null;
}

fn name(arena: std.mem.Allocator, io: std.Io, gpu: Gpu) ![]const u8 {
    if (std.process.run(arena, io, .{
        .argv = &.{ "lspci", "-vmm", "-s", &gpu.address },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(1024),
    })) |result| {
        if (result.term == .exited and result.term.exited == 0) {
            if (try lspciName(arena, result.stdout)) |lspci| return lspci;
        }
    } else |_| {}

    var buffer: [16]u8 = undefined;
    return std.fmt.allocPrint(arena, "Intel GPU {s}", .{gpu.dir.read("device", &buffer) catch "unknown"});
}

/// The vendor and device of `lspci -vmm`, unless pci.ids does not name the device.
fn lspciName(arena: std.mem.Allocator, output: []const u8) !?[]const u8 {
    var vendor: []const u8 = "";
    var device: []const u8 = "";
    var lines = std.mem.tokenizeScalar(u8, output, '\n');

    while (lines.next()) |line| {
        if (smi.sysfs.fieldValue(line, "Vendor")) |value| vendor = value;
        if (smi.sysfs.fieldValue(line, "Device")) |value| device = value;
    }

    if (device.len == 0 or std.mem.startsWith(u8, device, "Device ")) {
        return null;
    }

    return try std.fmt.allocPrint(arena, "{s} {s}", .{ vendor, device });
}

test lspciName {
    const b70 = (try lspciName(std.testing.allocator, "Slot:\tf3:00.0\nClass:\tVGA compatible controller\nVendor:\tIntel Corporation\nDevice:\tBattlemage G31 [Arc Pro B70]\nSVendor:\tIntel Corporation\nSDevice:\tDevice 1701\n")).?;
    defer std.testing.allocator.free(b70);
    try std.testing.expectEqualStrings("Intel Corporation Battlemage G31 [Arc Pro B70]", b70);
    try std.testing.expectEqual(null, try lspciName(std.testing.allocator, "Vendor:\tIntel Corporation\nDevice:\tDevice e223\n"));
}

test memoryBytes {
    const xe = smi.fdinfo.parse("drm-total-system:\t184 MiB\ndrm-resident-system:\t184 MiB\ndrm-total-vram0:\t4096 KiB\ndrm-resident-vram0:\t3072 KiB\n");
    try std.testing.expectEqual(3072 * 1024, memoryBytes(&xe));
    const evicted = smi.fdinfo.parse("drm-resident-system:\t2 MiB\ndrm-resident-vram0:\t0\n");
    try std.testing.expectEqual(0, memoryBytes(&evicted));
    const integrated = smi.fdinfo.parse("drm-total-system0:\t2 MiB\ndrm-resident-system0:\t2 MiB\n");
    try std.testing.expectEqual(2 * 1024 * 1024, memoryBytes(&integrated));
}
