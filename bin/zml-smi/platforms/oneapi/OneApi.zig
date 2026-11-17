//! Intel GPUs, through sysfs: drm, hwmon, and the fdinfo of every DRM client.
const std = @import("std");
const builtin = @import("builtin");
const model = @import("zml-smi/model");
const native = @import("zml-smi/native");
const source = @import("zml-smi/source");
const telemetry = @import("zml-smi/telemetry");

const OneApi = @This();

devices: []const model.Device,
io: std.Io,
gpus: []Gpu,
proc: ?native.sysfs.Dir,
/// Busy nanoseconds (i915), or busy and total cycles (xe).
counters: native.Counters(CounterKey),

const Gpu = struct {
    /// The PCI device, /sys/class/drm/renderD<N>/device.
    dir: native.sysfs.Dir,
    /// The device or the bridge above it that carries its link.
    link: native.sysfs.Dir,
    address: [12]u8,
    hwmon: ?Hwmon,
    energy: native.Counter = .{},
    mem_total_bytes: ?u64,

    fn close(gpu: Gpu) void {
        gpu.dir.close();
        gpu.link.close();
        if (gpu.hwmon) |hwmon| hwmon.dir.close();
    }
};

/// The files chosen in init, read every poll.
const Hwmon = struct {
    dir: native.sysfs.Dir,
    temperature: ?[]const u8,
    energy: ?[]const u8,
    power_limit: ?[]const u8,
    fan: ?Fan,
};

const pollSensors = telemetry.Bind(model.Gpu, .{
    .read = native.hwmon.readScalar,
    .fields = .{
        .temperature = .{ .temperature, .milli, .one },
        .power_limit_mw = .{ .power_limit, .micro, .milli },
    },
}).poll;

/// A fan PWM, else a fan speed and its maximum.
const Fan = struct {
    input: []const u8,
    /// Null for PWM; otherwise the file containing the maximum RPM.
    maximum: ?[]const u8 = null,

    /// The first PWM that reads, else the first speed that reads with a maximum above 0.
    fn find(arena: std.mem.Allocator, dir: native.sysfs.Dir) ?Fan {
        var input: [16]u8 = undefined;
        var max: [16]u8 = undefined;
        for (1..16) |index| {
            const pwm = std.fmt.bufPrint(&input, "pwm{d}", .{index}) catch unreachable;
            _ = dir.int(u64, pwm) catch continue;
            return .{ .input = arena.dupe(u8, pwm) catch return null };
        }
        for (1..16) |index| {
            const speed = std.fmt.bufPrint(&input, "fan{d}_input", .{index}) catch unreachable;
            const limit = std.fmt.bufPrint(&max, "fan{d}_max", .{index}) catch unreachable;
            _ = dir.int(u64, speed) catch continue;
            if ((dir.int(u64, limit) catch 0) == 0) continue;
            return .{
                .input = arena.dupe(u8, speed) catch return null,
                .maximum = arena.dupe(u8, limit) catch return null,
            };
        }
        return null;
    }

    fn percent(fan: Fan, dir: native.sysfs.Dir) ?u64 {
        const input = dir.int(u64, fan.input) catch return null;
        const file = fan.maximum orelse return native.hwmon.pwmPercent(input);
        const maximum = dir.int(u64, file) catch return null;
        return if (maximum > 0) @min(input * 100 / maximum, 100) else null;
    }
};

const EngineKey = struct { pid: u32, device: u16, class: u64 };
const CounterKey = struct { engine: EngineKey, metric: enum { busy, cycles } };

pub const two_readings = true;

const intel = 0x8086;

pub fn detect(io: std.Io) bool {
    return native.sysfs.exists(io, "/sys/class/drm");
}

pub fn open(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !OneApi {
    const drm = try native.sysfs.Dir.open(io, "/sys/class/drm");
    defer drm.close();

    var nodes: std.ArrayList(u32) = .empty;
    var it = drm.iterate();
    while (try it.next(io)) |entry| {
        const suffix = std.mem.cutPrefix(u8, entry.name, "renderD") orelse continue;
        const node = std.fmt.parseInt(u32, suffix, 10) catch continue;
        try nodes.append(arena, node);
    }
    std.mem.sort(u32, nodes.items, {}, std.sort.asc(u32));

    var driver_version: ?[]const u8 = null;
    var gpus: std.ArrayList(Gpu) = try .initCapacity(arena, nodes.items.len);
    errdefer for (gpus.items) |gpu| gpu.close();
    var devices: std.ArrayList(model.Device) = try .initCapacity(arena, nodes.items.len);
    for (nodes.items) |node| {
        const gpu = openGpu(arena, drm, node) catch continue;
        if (gpus.items.len == 0) driver_version = driverVersion(arena, io);
        gpus.appendAssumeCapacity(gpu);
        devices.appendAssumeCapacity(.{ .target = .oneapi, .name = try name(arena, io, gpu), .driver_version = driver_version });
    }

    return .{
        .devices = devices.items,
        .io = io,
        .gpus = gpus.items,
        .proc = native.sysfs.Dir.open(io, "/proc") catch null,
        .counters = .init(gpa),
    };
}

fn openGpu(arena: std.mem.Allocator, drm: native.sysfs.Dir, node: u32) !Gpu {
    var path: [32]u8 = undefined;
    const dir = try drm.openDir(try std.fmt.bufPrint(&path, "renderD{d}/device", .{node}));
    errdefer dir.close();
    if (try dir.int(u16, "vendor") != intel) return error.NotIntel;

    var buffer: [2048]u8 = undefined;
    const slot = try dir.field("uevent", "PCI_SLOT_NAME", &buffer);
    var address: [12]u8 = undefined;
    if (slot.len != address.len) return error.NotFound;
    @memcpy(&address, slot);

    return .{
        .dir = dir,
        .link = try native.pci.upstream(dir),
        .address = address,
        .hwmon = openHwmon(arena, dir),
        .mem_total_bytes = largestBar(dir.read("resource", &buffer) catch ""),
    };
}

fn openHwmon(arena: std.mem.Allocator, device: native.sysfs.Dir) ?Hwmon {
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const driver = device.readLink("driver", &buffer) catch null;
    const dir = native.hwmon.find(device, if (driver) |path| std.fs.path.basename(path) else null) catch return null;
    return .{
        .dir = dir,
        .temperature = native.hwmon.resolve(arena, dir, &.{
            .{ .prefix = "temp", .label = "pkg" },
            .{ .prefix = "temp", .label = "gpu" },
            .{ .prefix = "temp2" },
            .{ .prefix = "temp1" },
        }),
        .energy = native.hwmon.resolve(arena, dir, &.{
            .{ .prefix = "energy", .label = "card" },
            .{ .prefix = "energy", .label = "pkg" },
            .{ .prefix = "energy1" },
        }),
        .power_limit = native.hwmon.resolve(arena, dir, &.{ .{ .prefix = "power1", .suffix = "cap" }, .{ .prefix = "power1", .suffix = "max" } }),
        .fan = .find(arena, dir),
    };
}

pub fn close(self: *OneApi) void {
    for (self.gpus) |gpu| gpu.close();
    if (self.proc) |proc| proc.close();
    self.counters.deinit();
}

pub fn poll(self: *OneApi, scratch: std.mem.Allocator, reading: *source.Reading) !void {
    for (self.gpus, reading.samples) |*gpu, *sample| {
        const fields = &sample.oneapi;

        fields.clock_graphics_mhz = gpu.dir.int(u64, "tile0/gt0/freq0/act_freq") catch gpu.dir.int(u64, "tile0/gt0/freq0/cur_freq") catch null;
        fields.clock_graphics_max_mhz = gpu.dir.int(u64, "tile0/gt0/freq0/max_freq") catch null;
        fields.mem_total_bytes = gpu.mem_total_bytes;

        const link = native.pci.link(gpu.link, .current);
        fields.pcie_link_gen = link.gen;
        fields.pcie_link_width = link.width;
        fields.pcie_bandwidth_mbps = link.mbps();

        const hwmon = gpu.hwmon orelse continue;
        pollSensors(fields, .{&hwmon});
        if (hwmon.energy) |file| {
            // Energy is a running total: its growth per second is microwatts.
            if (hwmon.dir.int(u64, file)) |microjoules| {
                if (gpu.energy.advance(self.io, microjoules)) |growth| fields.power_mw = native.units.scale(growth.perSecond(), .micro, .milli);
            } else |_| {}
        }
        if (hwmon.fan) |fan| fields.fan_speed_percent = fan.percent(hwmon.dir);
    }

    // Read each DRM client once; its observation supplies both totals and process rows.
    defer self.counters.sweep();
    const proc = self.proc orelse return error.Unavailable;
    const devices = try scratch.alloc(DeviceUsage, self.gpus.len);
    @memset(devices, .{});

    var usages: std.AutoArrayHashMapUnmanaged(struct { u32, u16 }, Usage) = .empty;
    var engines: std.AutoArrayHashMapUnmanaged(EngineKey, Engine) = .empty;
    // A client shared through dup() or fork() is counted once.
    var clients: std.AutoHashMapUnmanaged(struct { u16, u64 }, void) = .empty;

    var holders: native.proc.FdHolders = .init(self.io, &.{"/dev/dri/"});
    defer holders.deinit();
    var buffer: [8192]u8 = undefined;
    while (holders.next()) |holder| {
        var path: [48]u8 = undefined;
        const fdinfo_path = std.fmt.bufPrint(&path, "{d}/fdinfo/{d}", .{ holder.pid, holder.fd }) catch continue;
        // The process may have closed it since: such a file is skipped.
        const info = native.fdinfo.read(proc, fdinfo_path, &buffer) catch continue;
        const device = self.deviceOf(info.pdev orelse continue) orelse continue;
        if (info.client_id) |id| {
            if ((try clients.getOrPut(scratch, .{ device, id })).found_existing) continue;
        }

        const usage = (try usages.getOrPutValue(scratch, .{ holder.pid, device }, .{})).value_ptr;
        if (memoryBytes(&info)) |bytes| {
            usage.mem_kib = (usage.mem_kib orelse 0) + bytes / 1024;
            devices[device].mem_kib += bytes / 1024;
        }
        for (info.engines()) |engine| {
            const xe = engine.total_cycles != null;
            const key: EngineKey = .{ .pid = holder.pid, .device = device, .class = if (xe) 0 else std.hash.Wyhash.hash(0, engine.class) };
            const totals = (try engines.getOrPutValue(scratch, key, .{ .kind = .of(engine.class) })).value_ptr;
            if (engine.total_cycles) |total| {
                totals.busy += engine.cycles orelse 0;
                totals.total = @max(totals.total orelse 0, total);
            } else {
                totals.busy += engine.busy_ns orelse 0;
            }
        }
    }

    for (engines.keys(), engines.values()) |key, engine| {
        const usage = usages.getPtr(.{ key.pid, key.device }).?;
        const device = &devices[key.device];
        if (!device.engines.contains(engine.kind)) device.engines.put(engine.kind, null);

        // Both counters advance at every reading, even the first.
        const busy = try self.counters.advance(.{ .engine = key, .metric = .busy }, self.io, engine.busy);
        const cycles = if (engine.total) |total|
            try self.counters.advance(.{ .engine = key, .metric = .cycles }, self.io, total)
        else
            null;
        const growth = busy orelse continue;
        if (engine.total != null and cycles == null) continue;
        native.Growth.add(&usage.busy, growth);
        native.Growth.add(&device.busy, growth);
        native.Growth.add(device.engines.getPtrAssertContains(engine.kind), growth);
        if (cycles) |total| {
            usage.cycles = total;
            if (device.cycles == null or total.amount > device.cycles.?.amount) device.cycles = total;
        }
    }

    for (reading.samples, devices) |*sample, device| {
        const fields = &sample.oneapi;
        fields.util_percent = percent(device.busy, device.cycles) orelse 0;
        fields.encoder_util_percent = if (device.engines.get(.encoder)) |growth| percent(growth, null) orelse 0 else null;
        fields.decoder_util_percent = if (device.engines.get(.decoder)) |growth| percent(growth, null) orelse 0 else null;
        fields.mem_used_bytes = device.mem_kib * 1024;
    }

    // Complete device totals before a process-list allocation can fail.
    for (usages.keys(), usages.values()) |key, usage| {
        try reading.addProcess(.{
            .pid = key[0],
            .device_idx = key[1],
            .dev_util_percent = percent(usage.busy, usage.cycles),
            .dev_mem_kib = usage.mem_kib,
        });
    }
}

/// One process on one device: its memory, and the growth of its busy time this round.
const Usage = struct {
    mem_kib: ?u64 = null,
    busy: ?native.Growth = null,
    /// xe: the growth of the total cycles the busy cycles are a share of.
    cycles: ?native.Growth = null,
};

/// The busy time of every process on one device.
const DeviceUsage = struct {
    mem_kib: u64 = 0,
    busy: ?native.Growth = null,
    cycles: ?native.Growth = null,
    /// A present key with no growth has been observed but needs another poll.
    engines: std.EnumMap(Engine.Kind, ?native.Growth) = .empty,
};

/// The busy time of one engine class, summed over the clients of a process.
const Engine = struct {
    busy: u64 = 0,
    /// xe: the total cycles the busy cycles are counted against.
    total: ?u64 = null,
    kind: Kind,

    /// Encoder and decoder utilization count only the classes named so.
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

/// Busy nanoseconds over the elapsed time (i915), or busy cycles over total cycles (xe).
fn percent(busy: ?native.Growth, cycles: ?native.Growth) ?u16 {
    const growth = busy orelse return null;
    return if (cycles) |total| growth.percentOf(total) else growth.percentOfTime();
}

fn deviceOf(self: *const OneApi, pdev: []const u8) ?u16 {
    for (self.gpus, 0..) |gpu, index| {
        if (std.mem.eql(u8, &gpu.address, pdev)) return @intCast(index);
    }
    return null;
}

/// The memory of one client: its resident VRAM, else its VRAM, else what it holds resident, else in total.
fn memoryBytes(info: *const native.fdinfo.Fdinfo) ?u64 {
    var sums: [2]struct { resident: u64 = 0, total: u64 = 0 } = @splat(.{});
    for (info.regions()) |region| {
        const sum = &sums[@intFromBool(!std.mem.startsWith(u8, region.name, "vram"))];
        sum.resident += region.resident orelse 0;
        sum.total += region.total orelse 0;
    }
    for (sums) |sum| {
        if (sum.resident > 0) return sum.resident;
        if (sum.total > 0) return sum.total;
    }
    return null;
}

/// Device memory: the largest BAR of the PCI resource file, if it is at least 256 MiB.
fn largestBar(resource: []const u8) ?u64 {
    var largest: u64 = 0;
    var lines = std.mem.tokenizeScalar(u8, resource, '\n');
    while (lines.next()) |line| {
        var words = std.mem.tokenizeAny(u8, line, " \t");
        const start = native.sysfs.parseInt(u64, words.next() orelse continue) catch continue;
        const end = native.sysfs.parseInt(u64, words.next() orelse continue) catch continue;
        if (end > start) largest = @max(largest, end - start + 1);
    }
    return if (largest >= 256 * 1024 * 1024) largest else null;
}

fn driverVersion(arena: std.mem.Allocator, io: std.Io) ?[]const u8 {
    const module = native.sysfs.Dir.open(io, "/sys/module") catch return null;
    defer module.close();
    var buffer: [128]u8 = undefined;
    for ([_][]const u8{ "xe/version", "xe/srcversion", "i915/version", "i915/srcversion" }) |file| {
        const version = module.read(file, &buffer) catch continue;
        if (version.len > 0) return arena.dupe(u8, version) catch null;
    }
    return null;
}

/// The name lspci gives, else the device id.
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

pub const testing = if (builtin.is_test) struct {
    pub const parseLspciName = OneApi.parseLspciName;
    pub const largestBar = OneApi.largestBar;
    pub const memoryBytes = OneApi.memoryBytes;
    pub const Fan = struct {
        pub const find = OneApi.Fan.find;
        pub const percent = OneApi.Fan.percent;
    };
} else struct {};
