const std = @import("std");
const sysfs = @import("zml-smi/sysfs");
const device_info = @import("zml-smi/info").device_info;
const GpuInfo = device_info.GpuInfo;
const DoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer;
const Collector = @import("zml-smi/collector").Collector;
const process = @import("process.zig");

pub fn start(collector: *Collector) !void {
    const poll_arena = try collector.createPollArena();
    const dev_offset: u16 = @intCast(collector.device_infos.items.len);

    var devices: std.ArrayList(Device) = .empty;
    var buffers: std.ArrayList(*DoubleBuffer(GpuInfo)) = .empty;
    var process_targets: std.ArrayList(process.Target) = .empty;

    for (0..64) |card_idx| {
        var dev = Device.open(collector.arena, poll_arena, collector.io, card_idx) catch continue;
        dev.device_idx = @intCast(dev_offset + devices.items.len);

        const initial = dev.static_info;
        const info = try collector.addDevice(.{ .oneapi = .{ .values = .{ initial, initial } } });

        try devices.append(collector.arena, dev);
        try buffers.append(collector.arena, &info.oneapi);
        try process_targets.append(collector.arena, .{
            .device_idx = dev.device_idx,
            .bdf = dev.bdf,
        });
    }

    if (devices.items.len == 0) return;

    try collector.spawnPoll(pollOnce, .{
        poll_arena,
        collector.io,
        try devices.toOwnedSlice(collector.arena),
        try buffers.toOwnedSlice(collector.arena),
    });

    const processes = try collector.createProcessList();
    try process.init(collector, processes, try process_targets.toOwnedSlice(collector.arena));
}

fn pollOnce(arena: *std.heap.ArenaAllocator, io: std.Io, devices: []Device, buffers: []*DoubleBuffer(GpuInfo)) void {
    poll(arena, io, devices, buffers) catch |err| {
        std.log.debug("oneapi poll skipped: {s}", .{@errorName(err)});
    };
}

fn poll(arena: *std.heap.ArenaAllocator, io: std.Io, devices: []Device, buffers: []*DoubleBuffer(GpuInfo)) !void {
    try pollSnapshot(arena, io, devices, buffers);
}

fn pollSnapshot(arena: *std.heap.ArenaAllocator, io: std.Io, devices: []Device, buffers: []*DoubleBuffer(GpuInfo)) !void {
    _ = arena.reset(.retain_capacity);
    var usage = process.collectDeviceUsage(arena.allocator(), io, processTargets(arena, devices)) catch std.AutoHashMapUnmanaged(u16, process.DeviceSample){};
    defer usage.deinit(arena.allocator());

    for (devices, buffers) |*dev, db| {
        const current = dev.snapshot(usage.get(dev.device_idx)) catch continue;
        defer dev.saveSnapshot(current);

        const back = db.back();
        back.* = dev.static_info;
        dev.fillMetrics(back, current);
        dev.fillDeltas(back, current);
        db.swap();
    }
}

fn processTargets(arena: *std.heap.ArenaAllocator, devices: []const Device) []process.Target {
    const targets = arena.allocator().alloc(process.Target, devices.len) catch return &.{};
    for (devices, targets) |dev, *target| {
        target.* = .{
            .device_idx = dev.device_idx,
            .bdf = dev.bdf,
        };
    }
    return targets;
}

const Device = struct {
    device_idx: u16 = 0,
    io: std.Io,
    arena: *std.heap.ArenaAllocator,
    card_path: []const u8,
    dev_path: []const u8,
    hwmon_path: ?[]const u8,
    static_info: GpuInfo,
    bdf: ?[process.bdf_len]u8,
    fdinfo_prev: ?process.EngineSample = null,
    energy_prev: ?EnergySample = null,

    fn open(allocator: std.mem.Allocator, scratch: *std.heap.ArenaAllocator, io: std.Io, card_idx: usize) !Device {
        const card_path = try std.fmt.allocPrint(allocator, "/sys/class/drm/card{d}", .{card_idx});
        errdefer allocator.free(card_path);
        const dev_path = try std.fmt.allocPrint(allocator, "{s}/device", .{card_path});
        errdefer allocator.free(dev_path);

        var vendor_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const vendor_path = try std.fmt.bufPrint(&vendor_path_buf, "{s}/vendor", .{dev_path});
        const vendor = try sysfs.readString(allocator, io, vendor_path);
        if (!std.mem.eql(u8, std.mem.trim(u8, vendor, &std.ascii.whitespace), "0x8086")) return error.not_intel;

        const hwmon_path = findHwmon(allocator, io, dev_path) catch null;
        var self: Device = .{
            .io = io,
            .arena = scratch,
            .card_path = card_path,
            .dev_path = dev_path,
            .hwmon_path = hwmon_path,
            .static_info = .{},
            .bdf = readBdf(allocator, io, dev_path) catch null,
        };
        self.static_info = .{
            .name = self.name(allocator, io) catch null,
            .driver_version = self.driverVersion(allocator, io) catch null,
            .power_limit_mw = self.powerLimit() catch null,
            .clock_graphics_max_mhz = self.maxClockGraphics() catch null,
            .mem_total_bytes = self.memTotal() catch null,
            .pcie_link_gen = self.pcieLinkGen() catch null,
            .pcie_link_width = self.pcieLinkWidth() catch null,
            .pcie_bandwidth_mbps = self.pcieBandwidth() catch null,
        };
        return self;
    }

    fn snapshot(self: Device, usage: ?process.DeviceSample) !Snapshot {
        return .{
            .fdinfo_activity = if (usage) |sample| sample.engine else null,
            .energy = self.energy() catch null,
            .mem_used_bytes = if (usage) |sample| sample.mem_kib * 1024 else 0,
        };
    }

    fn fillMetrics(self: Device, info: *GpuInfo, snap: Snapshot) void {
        inline for (metrics) |m| {
            @field(info, m.field) = m.query(self) catch null;
        }
        info.mem_used_bytes = snap.mem_used_bytes;
    }

    fn fillDeltas(self: Device, info: *GpuInfo, current: Snapshot) void {
        info.util_percent = fdinfoUtilization(self.fdinfo_prev, current.fdinfo_activity) orelse 0;
        info.power_mw = powerMilliwatts(self.energy_prev, current.energy);
    }

    fn saveSnapshot(self: *Device, current: Snapshot) void {
        self.fdinfo_prev = current.fdinfo_activity;
        self.energy_prev = current.energy;
    }

    fn name(self: Device, allocator: std.mem.Allocator, io: std.Io) ![]const u8 {
        const id = try self.deviceId(allocator, io);
        if (std.ascii.eqlIgnoreCase(id, "0xe223")) {
            return try allocator.dupe(u8, "Intel(R) Arc(TM) Pro B70 Graphics");
        }
        return try std.fmt.allocPrint(allocator, "Intel GPU {s}", .{id});
    }

    fn driverVersion(self: Device, allocator: std.mem.Allocator, io: std.Io) ![]const u8 {
        _ = self;
        const version = sysfs.readString(allocator, io, "/sys/module/xe/version") catch
            sysfs.readString(allocator, io, "/sys/module/xe/srcversion") catch
            sysfs.readString(allocator, io, "/sys/module/i915/version") catch
            try sysfs.readString(allocator, io, "/sys/module/i915/srcversion");
        const trimmed = std.mem.trim(u8, version, &std.ascii.whitespace);
        if (trimmed.len == 0) return error.not_found;
        return trimmed;
    }

    fn deviceId(self: Device, allocator: std.mem.Allocator, io: std.Io) ![]const u8 {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/device", .{self.dev_path});
        const raw = try sysfs.readString(allocator, io, path);
        return std.mem.trim(u8, raw, &std.ascii.whitespace);
    }

    pub fn temperature(self: Device) !u64 {
        const allocator = self.arena.allocator();
        const hwmon = self.hwmon_path orelse return error.not_found;
        const milli_c = readHwmonInput(allocator, self.io, hwmon, "temp", "pkg") catch
            readHwmonInput(allocator, self.io, hwmon, "temp", "gpu") catch
            readNumberedInput(allocator, self.io, hwmon, "temp", 2) catch
            try readNumberedInput(allocator, self.io, hwmon, "temp", 1);
        return (milli_c + 500) / 1000;
    }

    fn energy(self: Device) !EnergySample {
        const allocator = self.arena.allocator();
        const hwmon = self.hwmon_path orelse return error.not_found;
        const micro_joules = readHwmonInput(allocator, self.io, hwmon, "energy", "card") catch
            readHwmonInput(allocator, self.io, hwmon, "energy", "pkg") catch
            try readNumberedInput(allocator, self.io, hwmon, "energy", 1);
        return .{
            .micro_joules = micro_joules,
            .timestamp_ns = @intCast(std.Io.Timestamp.now(self.io, .awake).nanoseconds),
        };
    }

    pub fn powerLimit(self: Device) !u64 {
        const allocator = self.arena.allocator();
        const hwmon = self.hwmon_path orelse return error.not_found;
        const microwatts = readNumberedSuffix(allocator, self.io, hwmon, "power", 1, "cap") catch
            try readNumberedSuffix(allocator, self.io, hwmon, "power", 1, "max");
        return microwatts / 1000;
    }

    pub fn clockGraphics(self: Device) !u64 {
        return self.readFreq("act_freq") catch self.readFreq("cur_freq");
    }

    pub fn maxClockGraphics(self: Device) !u64 {
        return self.readFreq("max_freq");
    }

    fn readFreq(self: Device, file: []const u8) !u64 {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/tile0/gt0/freq0/{s}", .{ self.dev_path, file });
        return sysfs.readInt(self.arena.allocator(), self.io, path);
    }

    pub fn memTotal(self: Device) !u64 {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/resource", .{self.dev_path});
        return readLargestResource(self.arena.allocator(), self.io, path);
    }

    pub fn pcieLinkGen(self: Device) !u64 {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/current_link_speed", .{self.dev_path});
        const raw = try sysfs.readString(self.arena.allocator(), self.io, path);
        return pcieGenFromSpeed(raw);
    }

    pub fn pcieLinkWidth(self: Device) !u64 {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/current_link_width", .{self.dev_path});
        return sysfs.readInt(self.arena.allocator(), self.io, path);
    }

    pub fn pcieBandwidth(self: Device) !u64 {
        const gen = try self.pcieLinkGen();
        const width = try self.pcieLinkWidth();
        return pcieBandwidthFromLink(gen, width) orelse error.not_found;
    }
};

const metrics = .{
    .{ .field = "power_limit_mw", .query = Device.powerLimit },
    .{ .field = "temperature", .query = Device.temperature },
    .{ .field = "clock_graphics_mhz", .query = Device.clockGraphics },
    .{ .field = "clock_graphics_max_mhz", .query = Device.maxClockGraphics },
    .{ .field = "mem_total_bytes", .query = Device.memTotal },
    .{ .field = "pcie_bandwidth_mbps", .query = Device.pcieBandwidth },
    .{ .field = "pcie_link_gen", .query = Device.pcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.pcieLinkWidth },
};

const Snapshot = struct {
    fdinfo_activity: ?process.EngineSample = null,
    energy: ?EnergySample = null,
    mem_used_bytes: ?u64 = null,
};

const EnergySample = struct {
    micro_joules: u64,
    timestamp_ns: u64,
};

fn fdinfoUtilization(previous: ?process.EngineSample, current: ?process.EngineSample) ?u64 {
    return if (process.processUtil(previous, current)) |util| util else null;
}

fn powerMilliwatts(previous: ?EnergySample, current: ?EnergySample) ?u64 {
    const prev = previous orelse return null;
    const cur = current orelse return null;
    return milliwattsFromEnergyDelta(prev.micro_joules, cur.micro_joules, prev.timestamp_ns, cur.timestamp_ns);
}

pub fn milliwattsFromEnergyDelta(prev_uj: u64, cur_uj: u64, prev_ns: u64, cur_ns: u64) ?u64 {
    if (cur_uj < prev_uj or cur_ns <= prev_ns) return null;
    const energy_delta = cur_uj - prev_uj;
    const time_delta = cur_ns - prev_ns;
    if (time_delta == 0) return null;
    return energy_delta * 1_000_000 / time_delta;
}

fn findHwmon(allocator: std.mem.Allocator, io: std.Io, dev_path: []const u8) !?[]const u8 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const hwmon_root = try std.fmt.bufPrint(&path_buf, "{s}/hwmon", .{dev_path});
    var dir = std.Io.Dir.openDirAbsolute(io, hwmon_root, .{ .iterate = true }) catch return null;
    defer dir.close(io);

    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (!std.mem.startsWith(u8, entry.name, "hwmon")) continue;
        const candidate = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ hwmon_root, entry.name });
        var name_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const name_path = try std.fmt.bufPrint(&name_buf, "{s}/name", .{candidate});
        const name = sysfs.readString(allocator, io, name_path) catch return candidate;
        const trimmed = std.mem.trim(u8, name, &std.ascii.whitespace);
        if (std.mem.eql(u8, trimmed, "xe") or std.mem.eql(u8, trimmed, "i915")) return candidate;
        allocator.free(candidate);
    }
    return null;
}

fn readBdf(allocator: std.mem.Allocator, io: std.Io, dev_path: []const u8) ![process.bdf_len]u8 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/uevent", .{dev_path});
    const raw = try sysfs.readFieldString(allocator, io, path, "PCI_SLOT_NAME=");
    const trimmed = std.mem.trim(u8, raw, &std.ascii.whitespace);
    if (trimmed.len != process.bdf_len) return error.not_found;
    var out: [process.bdf_len]u8 = undefined;
    @memcpy(&out, trimmed);
    return out;
}

fn readHwmonInput(allocator: std.mem.Allocator, io: std.Io, hwmon: []const u8, comptime prefix: []const u8, label: []const u8) !u64 {
    for (1..16) |idx| {
        var label_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const label_path = try std.fmt.bufPrint(&label_path_buf, "{s}/{s}{d}_label", .{ hwmon, prefix, idx });
        const raw = sysfs.readString(allocator, io, label_path) catch continue;
        const trimmed = std.mem.trim(u8, raw, &std.ascii.whitespace);
        if (!std.ascii.eqlIgnoreCase(trimmed, label)) continue;
        return readNumberedInput(allocator, io, hwmon, prefix, idx);
    }
    return error.not_found;
}

fn readNumberedInput(allocator: std.mem.Allocator, io: std.Io, hwmon: []const u8, comptime prefix: []const u8, idx: usize) !u64 {
    return readNumberedSuffix(allocator, io, hwmon, prefix, idx, "input");
}

fn readNumberedSuffix(allocator: std.mem.Allocator, io: std.Io, hwmon: []const u8, comptime prefix: []const u8, idx: usize, suffix: []const u8) !u64 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}{d}_{s}", .{ hwmon, prefix, idx, suffix });
    return sysfs.readInt(allocator, io, path);
}

fn readLargestResource(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !u64 {
    var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    var line_writer: std.Io.Writer.Allocating = .init(allocator);
    defer line_writer.deinit();

    var largest: u64 = 0;
    while (true) {
        line_writer.clearRetainingCapacity();
        _ = reader.interface.streamDelimiter(&line_writer.writer, '\n') catch |err| switch (err) {
            error.EndOfStream => break,
            else => |e| return e,
        };
        reader.interface.toss(1);
        largest = @max(largest, resourceLineSize(line_writer.written()) orelse 0);
    }
    if (largest == 0) return error.not_found;
    return largest;
}

fn resourceLineSize(line: []const u8) ?u64 {
    var it = std.mem.tokenizeAny(u8, line, " \t");
    const base = parseHex(it.next() orelse return null) orelse return null;
    const end = parseHex(it.next() orelse return null) orelse return null;
    if (end < base) return null;
    const size = end - base + 1;
    return if (size >= 256 * 1024 * 1024) size else null;
}

fn parseHex(raw: []const u8) ?u64 {
    const trimmed = std.mem.trim(u8, raw, &std.ascii.whitespace);
    const digits = if (std.mem.startsWith(u8, trimmed, "0x")) trimmed[2..] else trimmed;
    return std.fmt.parseInt(u64, digits, 16) catch null;
}

fn pcieGenFromSpeed(raw: []const u8) !u64 {
    const trimmed = std.mem.trim(u8, raw, &std.ascii.whitespace);
    if (std.mem.startsWith(u8, trimmed, "2.5")) return 1;
    if (std.mem.startsWith(u8, trimmed, "5.0") or std.mem.startsWith(u8, trimmed, "5 ")) return 2;
    if (std.mem.startsWith(u8, trimmed, "8.0") or std.mem.startsWith(u8, trimmed, "8 ")) return 3;
    if (std.mem.startsWith(u8, trimmed, "16.0") or std.mem.startsWith(u8, trimmed, "16 ")) return 4;
    if (std.mem.startsWith(u8, trimmed, "32.0") or std.mem.startsWith(u8, trimmed, "32 ")) return 5;
    if (std.mem.startsWith(u8, trimmed, "64.0") or std.mem.startsWith(u8, trimmed, "64 ")) return 6;
    return error.not_found;
}

fn pcieBandwidthFromLink(gen: u64, width: u64) ?u64 {
    const lane_mb_s: u64 = switch (gen) {
        1 => 250,
        2 => 500,
        3 => 985,
        4 => 1969,
        5 => 3938,
        6 => 7563,
        else => return null,
    };
    return lane_mb_s * width;
}

test "oneAPI fdinfo utilization fallback" {
    try std.testing.expectEqual(@as(?u64, 50), fdinfoUtilization(.{ .engine_ns = 100, .timestamp_ns = 1000 }, .{ .engine_ns = 600, .timestamp_ns = 2000 }));
    try std.testing.expectEqual(@as(?u64, null), fdinfoUtilization(null, .{ .engine_ns = 600, .timestamp_ns = 2000 }));
}

test "oneAPI power delta from hwmon energy" {
    try std.testing.expectEqual(@as(?u64, 20_000), milliwattsFromEnergyDelta(1_000, 11_000, 1_000, 501_000));
    try std.testing.expectEqual(@as(?u64, null), milliwattsFromEnergyDelta(11_000, 1_000, 1_000, 501_000));
    try std.testing.expectEqual(@as(?u64, null), milliwattsFromEnergyDelta(1_000, 11_000, 501_000, 1_000));
}

test "oneAPI resource line chooses large memory BARs" {
    try std.testing.expectEqual(@as(?u64, 34_359_738_368), resourceLineSize("0x0000048800000000 0x0000048fffffffff 0x000000000014220c"));
    try std.testing.expectEqual(@as(?u64, null), resourceLineSize("0x00000000f6c00000 0x00000000f6dfffff 0x0000000000046200"));
}

test "oneAPI PCIe speed maps to generation" {
    try std.testing.expectEqual(@as(u64, 1), try pcieGenFromSpeed("2.5 GT/s PCIe"));
    try std.testing.expectEqual(@as(u64, 4), try pcieGenFromSpeed("16.0 GT/s PCIe"));
    try std.testing.expectError(error.not_found, pcieGenFromSpeed("Unknown"));
}

test "oneAPI PCIe bandwidth derives from generation and width" {
    try std.testing.expectEqual(@as(?u64, 250), pcieBandwidthFromLink(1, 1));
    try std.testing.expectEqual(@as(?u64, 63_008), pcieBandwidthFromLink(5, 16));
    try std.testing.expectEqual(@as(?u64, null), pcieBandwidthFromLink(0, 16));
}
