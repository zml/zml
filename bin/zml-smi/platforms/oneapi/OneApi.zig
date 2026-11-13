const std = @import("std");
const smi = @import("zml-smi");

const OneApi = @This();

platform: smi.Platform,
io: std.Io,
gpus: []Gpu,
proc: ?smi.sysfs.Dir,
busy: smi.Counters(EngineKey),
cycles: smi.Counters(EngineKey),
rows: ?[]const smi.ProcessInfo = null,

const Gpu = struct {
    dir: smi.sysfs.Dir,
    link: smi.sysfs.Dir,
    address: [12]u8,
    hwmon: ?Hwmon,
    energy: smi.Counter = .{},
    mem_total_bytes: ?u64,
};

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

    fn find(arena: std.mem.Allocator, dir: smi.sysfs.Dir) ?Fan {
        var input: [16]u8 = undefined;
        var max: [16]u8 = undefined;
        for (1..16) |index| {
            const pwm = std.fmt.bufPrint(&input, "pwm{d}", .{index}) catch unreachable;
            _ = dir.int(u64, pwm) catch continue;
            return .{ .pwm = arena.dupe(u8, pwm) catch return null };
        }
        for (1..16) |index| {
            const speed = std.fmt.bufPrint(&input, "fan{d}_input", .{index}) catch unreachable;
            const limit = std.fmt.bufPrint(&max, "fan{d}_max", .{index}) catch unreachable;
            _ = dir.int(u64, speed) catch continue;
            if ((dir.int(u64, limit) catch 0) == 0) continue;
            return .{ .speed = .{ .input = arena.dupe(u8, speed) catch return null, .max = arena.dupe(u8, limit) catch return null } };
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

const EngineKey = struct { pid: u32, device: u16, class: u64 };

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes, .deinit = deinit };

const intel = 0x8086;

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
    var devices: std.ArrayList(smi.Device) = .empty;
    for (nodes.items) |node| {
        const dir = openDevice(drm, node) catch continue;
        const gpu = openGpu(arena, dir) catch {
            dir.close();
            continue;
        };
        try gpus.append(arena, gpu);
        try devices.append(arena, .{ .target = .oneapi, .name = try name(arena, io, gpu), .driver_version = driver_version });
    }

    return .{
        .platform = .{ .devices = devices.items, .two_readings = true, .vtable = &vtable },
        .io = io,
        .gpus = gpus.items,
        .proc = smi.sysfs.Dir.open(io, "/proc") catch null,
        .busy = .init(gpa),
        .cycles = .init(gpa),
    };
}

fn renderNode(entry: []const u8) ?u32 {
    if (!std.mem.startsWith(u8, entry, "renderD")) return null;
    return std.fmt.parseInt(u32, entry["renderD".len..], 10) catch null;
}

fn openDevice(drm: smi.sysfs.Dir, node: u32) !smi.sysfs.Dir {
    var path: [32]u8 = undefined;
    return drm.openDir(try std.fmt.bufPrint(&path, "renderD{d}/device", .{node}));
}

fn openGpu(arena: std.mem.Allocator, dir: smi.sysfs.Dir) !Gpu {
    if (try dir.int(u16, "vendor") != intel) return error.NotIntel;

    var buffer: [2048]u8 = undefined;
    const slot = try dir.field("uevent", "PCI_SLOT_NAME", &buffer);
    var address: [12]u8 = undefined;
    if (slot.len != address.len) return error.NotFound;
    @memcpy(&address, slot);

    return .{
        .dir = dir,
        .link = try smi.pci.upstream(dir),
        .address = address,
        .hwmon = openHwmon(arena, dir),
        .mem_total_bytes = largestBar(dir.read("resource", &buffer) catch ""),
    };
}

fn openHwmon(arena: std.mem.Allocator, device: smi.sysfs.Dir) ?Hwmon {
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const driver = device.readLink("driver", &buffer) catch null;
    const dir = smi.hwmon.find(device, if (driver) |path| std.fs.path.basename(path) else null) catch return null;
    return .{
        .dir = dir,
        .temperature = smi.hwmon.resolve(arena, dir, &.{
            .{ .prefix = "temp", .label = "pkg" },
            .{ .prefix = "temp", .label = "gpu" },
            .{ .prefix = "temp2" },
            .{ .prefix = "temp1" },
        }),
        .energy = smi.hwmon.resolve(arena, dir, &.{
            .{ .prefix = "energy", .label = "card" },
            .{ .prefix = "energy", .label = "pkg" },
            .{ .prefix = "energy1" },
        }),
        .power_limit = smi.hwmon.resolve(arena, dir, &.{ .{ .prefix = "power1", .suffix = "cap" }, .{ .prefix = "power1", .suffix = "max" } }),
        .fan = .find(arena, dir),
    };
}

fn deinit(platform: *smi.Platform) void {
    const self: *OneApi = @alignCast(@fieldParentPtr("platform", platform));
    for (self.gpus) |gpu| {
        gpu.dir.close();
        gpu.link.close();
        if (gpu.hwmon) |hwmon| hwmon.dir.close();
    }
    if (self.proc) |proc| proc.close();
    self.busy.deinit();
    self.cycles.deinit();
}

fn poll(platform: *smi.Platform, scratch: std.mem.Allocator, samples: []smi.Sample) void {
    const self: *OneApi = @alignCast(@fieldParentPtr("platform", platform));

    for (self.gpus, samples) |*gpu, *sample| {
        const fields = &sample.oneapi;

        fields.clock_graphics_mhz = gpu.dir.int(u64, "tile0/gt0/freq0/act_freq") catch gpu.dir.int(u64, "tile0/gt0/freq0/cur_freq") catch null;
        fields.clock_graphics_max_mhz = gpu.dir.int(u64, "tile0/gt0/freq0/max_freq") catch null;
        fields.mem_total_bytes = gpu.mem_total_bytes;

        const link = smi.pci.link(gpu.link, .current);
        fields.pcie_link_gen = link.gen;
        fields.pcie_link_width = link.width;
        fields.pcie_bandwidth_mbps = link.mbps();

        const hwmon = gpu.hwmon orelse continue;
        if (hwmon.temperature) |file| fields.temperature = hwmon.dir.scaled(file, .milli, .one) catch null;
        if (hwmon.power_limit) |file| fields.power_limit_mw = hwmon.dir.scaled(file, .micro, .milli) catch null;
        if (hwmon.energy) |file| {
            if (hwmon.dir.int(u64, file)) |microjoules| {
                if (gpu.energy.advance(self.io, microjoules)) |growth| fields.power_mw = smi.units.scale(growth.perSecond(), .micro, .milli);
            } else |_| {}
        }
        if (hwmon.fan) |fan| fields.fan_speed_percent = fan.percent(hwmon.dir);
    }

    self.scan(scratch, samples);
}

fn processes(platform: *smi.Platform, _: std.mem.Allocator, _: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
    const self: *OneApi = @alignCast(@fieldParentPtr("platform", platform));
    for (self.rows orelse return error.Unavailable) |row| try out.append(row);
}

const Usage = struct {
    row: smi.ProcessInfo,
    busy: ?smi.Growth = null,
    cycles: ?smi.Growth = null,
};

const DeviceUsage = struct {
    busy: ?smi.Growth = null,
    cycles: ?smi.Growth = null,
    encoder: ?smi.Growth = null,
    decoder: ?smi.Growth = null,
    has_encoder: bool = false,
    has_decoder: bool = false,
};

fn scan(self: *OneApi, scratch: std.mem.Allocator, samples: []smi.Sample) void {
    defer self.busy.sweep();
    defer self.cycles.sweep();
    self.rows = null;
    const proc = self.proc orelse return;

    var usages: std.AutoArrayHashMapUnmanaged(struct { u32, u16 }, Usage) = .empty;
    var engines: std.AutoArrayHashMapUnmanaged(EngineKey, Engine) = .empty;
    // A client shared through dup() or fork() is counted once.
    var clients: std.AutoHashMapUnmanaged(struct { u16, u64 }, void) = .empty;

    var holders: smi.proc.FdHolders = .init(self.io, &.{"/dev/dri/"});
    defer holders.deinit();
    var buffer: [8192]u8 = undefined;
    while (holders.next()) |holder| {
        var path: [48]u8 = undefined;
        const fdinfo_path = std.fmt.bufPrint(&path, "{d}/fdinfo/{d}", .{ holder.pid, holder.fd }) catch continue;
        const info = smi.fdinfo.read(proc, fdinfo_path, &buffer) catch continue;
        const device = self.deviceOf(info.pdev orelse continue) orelse continue;
        if (info.client_id) |id| {
            if ((clients.getOrPut(scratch, .{ device, id }) catch continue).found_existing) continue;
        }

        const usage = usages.getOrPutValue(scratch, .{ holder.pid, device }, .{ .row = .{ .pid = holder.pid, .device_idx = device } }) catch continue;
        if (memoryBytes(&info)) |bytes| usage.value_ptr.row.dev_mem_kib = (usage.value_ptr.row.dev_mem_kib orelse 0) + bytes / 1024;
        for (info.engines()) |engine| {
            const xe = engine.total_cycles != null;
            const key: EngineKey = .{ .pid = holder.pid, .device = device, .class = if (xe) 0 else std.hash.Wyhash.hash(0, engine.class) };
            const totals = (engines.getOrPutValue(scratch, key, .{ .kind = .of(engine.class) }) catch continue).value_ptr;
            if (engine.total_cycles) |total| {
                totals.busy += engine.cycles orelse 0;
                totals.total = @max(totals.total orelse 0, total);
            } else {
                totals.busy += engine.busy_ns orelse 0;
            }
        }
    }

    const devices = scratch.alloc(DeviceUsage, self.gpus.len) catch return;
    @memset(devices, .{});
    var it = engines.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const usage = usages.getPtr(.{ key.pid, key.device }).?;
        const device = &devices[key.device];
        device.has_encoder = device.has_encoder or entry.value_ptr.kind == .encoder;
        device.has_decoder = device.has_decoder or entry.value_ptr.kind == .decoder;

        // Both counters advance at every reading, even the first.
        const busy = self.busy.advance(key, self.io, entry.value_ptr.busy) catch continue;
        const cycles = if (entry.value_ptr.total) |total| self.cycles.advance(key, self.io, total) catch continue else null;
        const growth = busy orelse continue;
        if (entry.value_ptr.total != null and cycles == null) continue;
        smi.Growth.add(&usage.busy, growth);
        smi.Growth.add(&device.busy, growth);
        switch (entry.value_ptr.kind) {
            .encoder => smi.Growth.add(&device.encoder, growth),
            .decoder => smi.Growth.add(&device.decoder, growth),
            .other => {},
        }
        if (cycles) |total| {
            usage.cycles = total;
            if (device.cycles == null or total.amount > device.cycles.?.amount) device.cycles = total;
        }
    }

    const rows = scratch.alloc(smi.ProcessInfo, usages.count()) catch return;
    for (rows, usages.values()) |*row, usage| {
        row.* = usage.row;
        row.dev_util_percent = percent(usage.busy, usage.cycles);
    }
    self.rows = rows;

    for (samples, devices, 0..) |*sample, device, index| {
        const fields = &sample.oneapi;
        fields.util_percent = percent(device.busy, device.cycles) orelse 0;
        fields.encoder_util_percent = if (device.has_encoder) percent(device.encoder, null) orelse 0 else null;
        fields.decoder_util_percent = if (device.has_decoder) percent(device.decoder, null) orelse 0 else null;
        var mem_kib: u64 = 0;
        for (rows) |row| {
            if (row.device_idx == index) mem_kib += row.dev_mem_kib orelse 0;
        }
        fields.mem_used_bytes = mem_kib * 1024;
    }
}

const Engine = struct {
    busy: u64 = 0,
    total: ?u64 = null,
    kind: Kind,

    const Kind = enum {
        encoder,
        decoder,
        other,

        fn of(class: []const u8) Kind {
            if (std.ascii.findIgnoreCase(class, "encode") != null) return .encoder;
            if (std.ascii.findIgnoreCase(class, "decode") != null) return .decoder;
            return .other;
        }
    };
};

fn percent(busy: ?smi.Growth, cycles: ?smi.Growth) ?u16 {
    const growth = busy orelse return null;
    return if (cycles) |total| growth.percentOf(total) else growth.percentOfTime();
}

fn deviceOf(self: *const OneApi, pdev: []const u8) ?u16 {
    for (self.gpus, 0..) |gpu, index| {
        if (std.mem.eql(u8, &gpu.address, pdev)) return @intCast(index);
    }
    return null;
}

fn memoryBytes(info: *const smi.fdinfo.Fdinfo) ?u64 {
    var sums: [4]u64 = @splat(0);
    for (info.regions()) |region| {
        if (std.mem.startsWith(u8, region.name, "vram")) {
            sums[0] += region.resident orelse 0;
            sums[1] += region.total orelse 0;
        }
        sums[2] += region.resident orelse 0;
        sums[3] += region.total orelse 0;
    }
    for (sums) |sum| {
        if (sum > 0) return sum;
    }
    return null;
}

/// Device memory: the largest BAR of the PCI resource file, if it is at least 256 MiB.
fn largestBar(resource: []const u8) ?u64 {
    var largest: u64 = 0;
    var lines = std.mem.tokenizeScalar(u8, resource, '\n');
    while (lines.next()) |line| {
        var words = std.mem.tokenizeAny(u8, line, " \t");
        const start = smi.sysfs.parseInt(u64, words.next() orelse continue) catch continue;
        const end = smi.sysfs.parseInt(u64, words.next() orelse continue) catch continue;
        if (end > start) largest = @max(largest, end - start + 1);
    }
    return if (largest >= 256 * 1024 * 1024) largest else null;
}

fn driverVersion(arena: std.mem.Allocator, io: std.Io) ?[]const u8 {
    const module = smi.sysfs.Dir.open(io, "/sys/module") catch return null;
    defer module.close();
    var buffer: [128]u8 = undefined;
    for ([_][]const u8{ "xe/version", "xe/srcversion", "i915/version", "i915/srcversion" }) |file| {
        const version = module.read(file, &buffer) catch continue;
        if (version.len > 0) return arena.dupe(u8, version) catch null;
    }
    return null;
}

fn name(arena: std.mem.Allocator, io: std.Io, gpu: Gpu) ![]const u8 {
    if (std.process.run(arena, io, .{
        .argv = &.{ "lspci", "-Dnn", "-s", &gpu.address },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(1024),
    })) |result| {
        if (result.term == .exited and result.term.exited == 0) {
            if (parseLspciName(result.stdout)) |lspci| return arena.dupe(u8, lspci);
        }
    } else |_| {}

    var buffer: [16]u8 = undefined;
    return std.fmt.allocPrint(arena, "Intel GPU {s}", .{gpu.dir.read("device", &buffer) catch "unknown"});
}

fn parseLspciName(output: []const u8) ?[]const u8 {
    var lines = std.mem.tokenizeScalar(u8, output, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
        const header_end = std.mem.indexOf(u8, trimmed, "]: ") orelse continue;
        const header = trimmed[0 .. header_end + 1];
        if (std.ascii.findIgnoreCase(header, "VGA") == null and std.ascii.findIgnoreCase(header, "Display") == null) continue;

        const rest = trimmed[header_end + "]: ".len ..];
        const end = std.mem.lastIndexOf(u8, rest, " [8086:") orelse std.mem.indexOf(u8, rest, " (rev ") orelse rest.len;
        const device_name = std.mem.trim(u8, rest[0..end], &std.ascii.whitespace);
        if (device_name.len == 0 or std.mem.startsWith(u8, device_name, "Intel Corporation Device ")) return null;
        return device_name;
    }
    return null;
}

test parseLspciName {
    try std.testing.expectEqualStrings(
        "Intel Corporation Battlemage G31 [Arc Pro B70]",
        parseLspciName("0000:84:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Arc Pro B70] [8086:e223] (rev 04)\n").?,
    );
    try std.testing.expectEqual(null, parseLspciName("0000:84:00.0 Display controller [0380]: Intel Corporation Device e223 [8086:e223] (rev 04)\n"));
}

test largestBar {
    try std.testing.expectEqual(34_359_738_368, largestBar(
        \\0x0000048800000000 0x0000048fffffffff 0x000000000014220c
        \\0x00000000f6c00000 0x00000000f6dfffff 0x0000000000046200
        \\0x0000000000000000 0x0000000000000000 0x0000000000000000
    ));
    try std.testing.expectEqual(null, largestBar("0x00000000f6c00000 0x00000000f6dfffff 0x0000000000046200\n"));
}

test memoryBytes {
    const xe = smi.fdinfo.parse("drm-total-system:\t184 MiB\ndrm-total-vram0:\t4096 KiB\ndrm-resident-vram0:\t3072 KiB\n");
    try std.testing.expectEqual(3072 * 1024, memoryBytes(&xe));
    const system = smi.fdinfo.parse("drm-total-system0:\t2 MiB\n");
    try std.testing.expectEqual(2 * 1024 * 1024, memoryBytes(&system));
}
