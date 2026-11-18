//! Intel GPUs, through sysfs: drm, hwmon, and the fdinfo of every DRM client.
const std = @import("std");
const builtin = @import("builtin");
const contract = @import("zml-smi-2/contract");
const native = @import("zml-smi-2/native");
const source = @import("zml-smi-2/source");
const polling = @import("zml-smi-2/poll");

const OneApi = @This();

devices: []const contract.Device,
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

const SensorProbe = struct {
    device: *Gpu,

    pub const temperature = native.hwmon.Scalar(SensorProbe, .{ .temperature, .milli, .one });
    pub const power_limit_mw = native.hwmon.Scalar(SensorProbe, .{ .power_limit, .micro, .milli });

    pub fn power_mw(self: *const SensorProbe) ?u64 {
        const hwmon = if (self.device.hwmon) |*sensors| sensors else return null;
        const file = hwmon.energy orelse return null;
        // Energy is a running total: its growth per second is microwatts.
        const microjoules = hwmon.dir.int(u64, file) catch return null;
        const growth = self.device.energy.advance(self.device.dir.io, microjoules) orelse return null;
        return native.units.scale(growth.perSecond(), .micro, .milli);
    }

    pub fn fan_speed_percent(self: *const SensorProbe) ?u64 {
        const hwmon = if (self.device.hwmon) |*sensors| sensors else return null;
        const fan = hwmon.fan orelse return null;
        return fan.percent(hwmon.dir);
    }
};

/// A fan PWM, else a fan speed and its maximum.
const Fan = struct {
    index: u4,
    pwm: bool,

    /// The first PWM that reads, else the first speed that reads with a maximum above 0.
    fn find(dir: native.sysfs.Dir) ?Fan {
        for ([_]bool{ true, false }) |pwm| {
            for (1..16) |index| {
                const fan: Fan = .{ .index = @intCast(index), .pwm = pwm };
                if (fan.percent(dir) != null) return fan;
            }
        }
        return null;
    }

    fn percent(fan: Fan, dir: native.sysfs.Dir) ?u64 {
        var path: [16]u8 = undefined;
        if (fan.pwm) return native.hwmon.pwmPercent(fan.read(dir, "pwm{d}", &path) orelse return null);
        const input = fan.read(dir, "fan{d}_input", &path) orelse return null;
        const maximum = fan.read(dir, "fan{d}_max", &path) orelse return null;
        return if (maximum > 0) @intCast(@min(@as(u128, input) * 100 / maximum, 100)) else null;
    }

    fn read(fan: Fan, dir: native.sysfs.Dir, comptime format: []const u8, path: *[16]u8) ?u64 {
        const file = std.fmt.bufPrint(path, format, .{fan.index}) catch unreachable;
        return dir.int(u64, file) catch null;
    }
};

const EngineKey = struct { pid: u32, device: u16, class: u64 };
const CounterKey = struct { engine: EngineKey, metric: enum { busy, cycles } };

pub const two_readings = true;

const intel = 0x8086;

pub const detection_paths = [_][]const u8{"/sys/class/drm"};

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
    var devices: std.ArrayList(contract.Device) = try .initCapacity(arena, nodes.items.len);
    for (nodes.items) |node| {
        const gpu = openGpu(arena, drm, node) catch continue;
        if (gpus.items.len == 0) driver_version = driverVersion(arena, io);
        gpus.appendAssumeCapacity(gpu);
        devices.appendAssumeCapacity(.{ .target = .oneapi, .name = try name(arena, io, gpu), .driver_version = driver_version });
    }

    return .{
        .devices = devices.items,
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
    if (slot.len != 12) return error.NotFound;

    return .{
        .dir = dir,
        .link = try native.pci.upstream(dir),
        .address = slot[0..12].*,
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
        .fan = .find(dir),
    };
}

pub fn close(self: *OneApi) void {
    for (self.gpus) |gpu| gpu.close();
    if (self.proc) |proc| proc.close();
    self.counters.deinit();
}

pub fn poll(self: *OneApi, scratch: std.mem.Allocator, reading: *source.Reading) !void {
    for (self.gpus, reading.samples) |*gpu, *sample| {
        const graphics = gpu.dir.int(u64, "tile0/gt0/freq0/act_freq") catch
            gpu.dir.int(u64, "tile0/gt0/freq0/cur_freq") catch null;
        sample.put(.clock_graphics_mhz, graphics);
        sample.put(.clock_graphics_max_mhz, gpu.dir.int(u64, "tile0/gt0/freq0/max_freq") catch null);
        sample.put(.mem_total_bytes, gpu.mem_total_bytes);

        const link = native.pci.link(gpu.link, .current);
        sample.put(.pcie_link_gen, link.gen);
        sample.put(.pcie_link_width, link.width);
        sample.put(.pcie_bandwidth_mbps, link.mbps());

        polling.project(sample, &SensorProbe{ .device = gpu });
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

    var holders: native.proc.FdHolders = .init(proc.io, &.{"/dev/dri/"});
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
            const kib = bytes / 1024;
            usage.mem_kib = (usage.mem_kib orelse 0) + kib;
            devices[device].total.mem_kib = (devices[device].total.mem_kib orelse 0) + kib;
        }
        for (info.engines.constSlice()) |engine| {
            const xe = engine.total_cycles != null;
            const key: EngineKey = .{ .pid = holder.pid, .device = device, .class = if (xe) 0 else std.hash.Wyhash.hash(0, engine.class) };
            const totals = (try engines.getOrPutValue(scratch, key, .{ .kind = .of(engine.class) })).value_ptr;
            totals.busy += (if (xe) engine.cycles else engine.busy_ns) orelse 0;
            if (engine.total_cycles) |total| totals.total = @max(totals.total orelse 0, total);
        }
    }

    for (engines.keys(), engines.values()) |key, engine| {
        const usage = usages.getPtr(.{ key.pid, key.device }).?;
        const device = &devices[key.device];
        if (!device.engines.contains(engine.kind)) device.engines.put(engine.kind, null);

        // Both counters advance at every reading, even the first.
        const busy = try self.counters.advance(.{ .engine = key, .metric = .busy }, proc.io, engine.busy);
        const cycles = if (engine.total) |total|
            try self.counters.advance(.{ .engine = key, .metric = .cycles }, proc.io, total)
        else
            null;
        const growth = busy orelse continue;
        if (engine.total != null and cycles == null) continue;
        native.Growth.add(&usage.busy, growth);
        native.Growth.add(&device.total.busy, growth);
        native.Growth.add(device.engines.getPtrAssertContains(engine.kind), growth);
        if (cycles) |total| {
            usage.cycles = total;
            if (device.total.cycles == null or total.amount > device.total.cycles.?.amount) {
                device.total.cycles = total;
            }
        }
    }

    for (reading.samples, devices) |*sample, *device| {
        sample.put(.util_percent, device.total.percent() orelse 0);
        sample.put(.encoder_util_percent, device.enginePercent(.encoder));
        sample.put(.decoder_util_percent, device.enginePercent(.decoder));
        sample.put(.mem_used_bytes, (device.total.mem_kib orelse 0) * 1024);
    }

    // Complete device totals before a process-list allocation can fail.
    for (usages.keys(), usages.values()) |key, usage| {
        try reading.addProcess(.{
            .pid = key[0],
            .device_idx = key[1],
            .dev_util_percent = usage.percent(),
            .dev_mem_kib = usage.mem_kib,
        });
    }
}

/// Memory and busy time growth this round, per process or summed for a device.
const Usage = struct {
    mem_kib: ?u64 = null,
    busy: ?native.Growth = null,
    /// xe: the growth of the total cycles the busy cycles are a share of.
    cycles: ?native.Growth = null,

    fn percent(usage: Usage) ?u16 {
        const growth = usage.busy orelse return null;
        return if (usage.cycles) |total| growth.percentOf(total) else growth.percentOfTime();
    }
};

/// The busy time of every process on one device.
const DeviceUsage = struct {
    total: Usage = .{},
    /// A present key with no growth has been observed but needs another poll.
    engines: std.EnumMap(Engine.Kind, ?native.Growth) = .empty,

    fn enginePercent(device: *const DeviceUsage, kind: Engine.Kind) ?u64 {
        const growth = device.engines.get(kind) orelse return null;
        return if (growth) |value| value.percentOfTime() else 0;
    }
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

fn deviceOf(self: *const OneApi, pdev: []const u8) ?u16 {
    for (self.gpus, 0..) |gpu, index| {
        if (std.mem.eql(u8, &gpu.address, pdev)) return @intCast(index);
    }
    return null;
}

/// The memory of one client: its resident VRAM, else its VRAM, else what it holds resident, else in total.
fn memoryBytes(info: *const native.fdinfo.Fdinfo) ?u64 {
    var sums: [2]struct { resident: u64 = 0, total: u64 = 0 } = @splat(.{});
    for (info.regions.constSlice()) |region| {
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
    const result = std.process.run(arena, io, .{
        .argv = &.{ "lspci", "-Dnn", "-s", &gpu.address },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(1024),
    }) catch null;
    if (result) |run| {
        if (run.term == .exited and run.term.exited == 0) {
            if (parseLspciName(run.stdout)) |lspci| return arena.dupe(u8, lspci);
        }
    }

    var buffer: [16]u8 = undefined;
    return std.fmt.allocPrint(arena, "Intel GPU {s}", .{gpu.dir.read("device", &buffer) catch "unknown"});
}

fn parseLspciName(output: []const u8) ?[]const u8 {
    var lines = std.mem.tokenizeScalar(u8, output, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
        const header, const rest = std.mem.cut(u8, trimmed, "]: ") orelse continue;
        if (std.ascii.findIgnoreCase(header, "VGA") == null and std.ascii.findIgnoreCase(header, "Display") == null) continue;

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
