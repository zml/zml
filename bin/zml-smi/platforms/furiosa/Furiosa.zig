//! FuriosaAI RNGD chips, through sysfs. Paths, units and formulas follow furiosa-smi-go 2026.4.0.
const std = @import("std");
const smi = @import("zml-smi");

const Furiosa = @This();

platform: smi.Platform,
io: std.Io,
class: smi.sysfs.Dir,
chips: []Chip,

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes, .deinit = deinit };

pub fn detect(io: std.Io) bool {
    return smi.sysfs.exists(io, "/dev/rngd");
}

pub fn init(arena: std.mem.Allocator, _: std.mem.Allocator, io: std.Io) !Furiosa {
    const usable = try usableCores(io);
    const class = try smi.sysfs.Dir.open(io, "/sys/class/rngd_mgmt");
    errdefer class.close();

    var chips: std.ArrayList(Chip) = .empty;
    errdefer for (chips.items) |chip| {
        chip.mgmt.close();
        chip.pci.close();
        if (chip.hwmon) |hwmon| hwmon.dir.close();
    };

    var it = class.iterate();
    while (try it.next(io)) |entry| {
        const node = mgmtNode(entry.name) orelse continue;

        // A chip with no node in /dev/rngd is not this host's.
        if (usable[node].count() == 0) continue;

        const mgmt = class.openDir(entry.name) catch continue;
        const chip = openChip(arena, mgmt, node, usable[node]) catch {
            mgmt.close();
            continue;
        };

        chips.append(arena, chip) catch |err| {
            chip.mgmt.close();
            chip.pci.close();
            if (chip.hwmon) |hwmon| hwmon.dir.close();
            return err;
        };
    }
    std.mem.sort(Chip, chips.items, {}, Chip.lessThan);

    const devices = try arena.alloc(smi.Device, chips.items.len);
    for (devices, chips.items) |*device, chip| {
        var buffer: [128]u8 = undefined;
        var details: std.ArrayList(smi.Device.Detail) = .empty;

        const fw = chip.mgmt.read("fw_version", &buffer) catch "";
        if (fw.len > 0) try details.append(arena, .{ .key = "firmware_version", .label = "Firmware", .value = try arena.dupe(u8, fw) });

        const sn = chip.mgmt.read("device_sn", &buffer) catch "";
        if (sn.len > 0) try details.append(arena, .{ .key = "serial", .label = "Serial", .value = try arena.dupe(u8, sn) });

        const version = chip.mgmt.read("version", &buffer) catch "";
        const driver_version = if (version.len > 0) try arena.dupe(u8, version) else null;

        device.* = .{
            .target = .furiosa,
            .name = chip.model.name,
            .driver_version = driver_version,
            .details = details.items,
        };
    }

    return .{
        .platform = .{ .devices = devices, .two_readings = true, .vtable = &vtable },
        .io = io,
        .class = class,
        .chips = chips.items,
    };
}

fn poll(platform: *smi.Platform, _: std.mem.Allocator, samples: []smi.Sample) void {
    const self: *Furiosa = @alignCast(@fieldParentPtr("platform", platform));

    for (self.chips, samples) |*chip, *sample| {
        const fields = &sample.furiosa;
        fields.mem_total_bytes = chip.model.memoryBytes() / cores * chip.usable.count();
        fields.clock_core_mhz = coreClock(chip.mgmt, chip.usable);
        fields.governor = governor(chip.mgmt);
        fields.liveness = liveness(chip.mgmt);
        fields.throttle_reasons = throttleReasons(chip.mgmt);

        const link = smi.pci.link(chip.pci, .current);
        fields.pcie_link_gen = link.gen;
        fields.pcie_link_width = link.width;
        fields.pcie_bandwidth_mbps = link.mbps();

        chip.util = @splat(0);
        chip.mem_used = @splat(0);
        if (occupancy(chip.mgmt)) |in_use| {
            fields.util_percent = self.utilization(chip, in_use);
            fields.mem_used_bytes = self.memoryInUse(chip, in_use);
        }

        const hwmon = chip.hwmon orelse continue;
        fields.temperature = milli(hwmon.dir, hwmon.temperature);
        fields.ambient_temperature = milli(hwmon.dir, hwmon.ambient);
        fields.power_mw = milli(hwmon.dir, hwmon.power);
        fields.power_limit_mw = milli(hwmon.dir, "power1_max");
    }
}

fn processes(platform: *smi.Platform, scratch: std.mem.Allocator, samples: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
    const self: *Furiosa = @alignCast(@fieldParentPtr("platform", platform));

    var held: std.AutoArrayHashMapUnmanaged(struct { u32, u16 }, Cores) = .empty;
    var holders: smi.procfs.FdHolders = .init(self.io, &.{"/dev/rngd/"});
    defer holders.deinit();

    while (holders.next()) |holder| {
        const node = Node.parse(holder.rest) orelse continue;
        const device = self.deviceOf(node.chip) orelse continue;
        const entry = try held.getOrPutValue(scratch, .{ holder.pid, device }, .empty);
        entry.value_ptr.setUnion(node.cores);
    }

    for (held.keys(), held.values()) |key, set| {
        const pid, const device = key;
        try out.append(self.chips[device].row(pid, device, set, samples[device].furiosa));
    }
}

fn deinit(platform: *smi.Platform) void {
    const self: *Furiosa = @alignCast(@fieldParentPtr("platform", platform));

    for (self.chips) |chip| {
        chip.mgmt.close();
        chip.pci.close();
        if (chip.hwmon) |hwmon| hwmon.dir.close();
    }

    self.class.close();
}

const cores = 8; // we'll see if new cards get released
const Cores = std.bit_set.Integer(cores);

const Chip = struct {
    node: u8,
    model: *const Model,
    mgmt: smi.sysfs.Dir,
    pci: smi.sysfs.Dir,
    hwmon: ?Hwmon,
    usable: Cores,
    cycles: [cores]?Cycles = @splat(null),
    util: [cores]u8 = @splat(0),
    mem_used: [cores]u64 = @splat(0),

    fn lessThan(_: void, a: Chip, b: Chip) bool {
        if (a.model.device != b.model.device) return a.model.device < b.model.device;
        return a.node < b.node;
    }

    fn row(chip: *const Chip, pid: u32, device: u16, held: Cores, sample: smi.Sample.Furiosa) smi.ProcessInfo {
        const mine = held.intersectWith(chip.usable);
        var util: u64 = 0;
        var mem_used: u64 = 0;
        var it = mine.iterator(.{});

        while (it.next()) |core| {
            util += chip.util[core];
            mem_used += chip.mem_used[core];
        }

        return .{
            .pid = pid,
            .device_idx = device,
            .dev_util_percent = if (sample.util_percent != null) @intCast(util / @max(@as(u64, mine.count()), 1)) else null,
            .dev_mem_kib = if (sample.mem_used_bytes != null) mem_used / 1024 else null,
        };
    }
};

const Cycles = struct { all: u64, task: u64 };

const Hwmon = struct {
    dir: smi.sysfs.Dir,
    temperature: ?[]const u8,
    ambient: ?[]const u8,
    power: ?[]const u8,

    fn open(arena: std.mem.Allocator, device: smi.sysfs.Dir) ?Hwmon {
        const dir = smi.hwmon.find(device, null) catch return null;
        return .{
            .dir = dir,
            .temperature = smi.hwmon.labeled(arena, dir, "temp", "input", "PEAK"),
            .ambient = smi.hwmon.labeled(arena, dir, "temp", "input", "AMBIENT"),
            .power = smi.hwmon.labeled(arena, dir, "power", "average", "RMS_TOTAL"),
        };
    }
};

/// A device type. Its firmware reserves 512 MiB of its DRAM, and its cores share the rest evenly.
const Model = struct {
    device: u16,
    name: []const u8,
    dram_bytes: u64,

    fn memoryBytes(model: Model) u64 {
        return model.dram_bytes - 512 * 1024 * 1024;
    }
};

const models = [_]Model{
    .{ .device = 0x0001, .name = "RNGD", .dram_bytes = 48 << 30 },
    .{ .device = 0x0002, .name = "RNGD+", .dram_bytes = 72 << 30 },
};

const vendor = 0x1ed2;

fn usableCores(io: std.Io) ![256]Cores {
    var usable: [256]Cores = @splat(.empty);
    const dev = try smi.sysfs.Dir.open(io, "/dev/rngd");
    defer dev.close();

    var it = dev.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .character_device) continue;
        const node = Node.parse(entry.name) orelse continue;
        usable[node.chip].setUnion(node.cores);
    }

    return usable;
}

fn openChip(arena: std.mem.Allocator, mgmt: smi.sysfs.Dir, node: u8, usable: Cores) !Chip {
    var buffer: [64]u8 = undefined;
    var path: [64]u8 = undefined;
    const pci = try smi.sysfs.Dir.open(mgmt.io, try std.fmt.bufPrint(&path, "/sys/bus/pci/devices/{s}", .{try mgmt.read("busname", &buffer)}));
    errdefer pci.close();

    if (try pci.int(u16, "vendor") != vendor) return error.NotFuriosa;

    const device = try pci.int(u16, "device");
    const model = for (&models) |*model| {
        if (model.device == device) break model;
    } else return error.UnknownDevice;

    return .{ .node = node, .model = model, .mgmt = mgmt, .pci = pci, .hwmon = .open(arena, pci), .usable = usable };
}

fn utilization(self: *const Furiosa, chip: *Chip, in_use: Cores) ?u64 {
    var sum: u64 = 0;
    var it = chip.usable.iterator(.{});

    while (it.next()) |core| {
        var buffer: [1024]u8 = undefined;
        const current: Cycles = blk: {
            if (!in_use.isSet(core)) break :blk .{ .all = 0, .task = 0 };
            const text = (self.coreFile(chip.node, core, "metrics", &buffer) catch return null) orelse break :blk .{ .all = 0, .task = 0 };

            break :blk .{
                .all = hexField(text, "CycleCount") orelse return null,
                .task = hexField(text, "TaskCycleCount") orelse return null,
            };
        };
        const prev = chip.cycles[core];
        chip.cycles[core] = current;

        chip.util[core] = blk: {
            const p = prev orelse break :blk 0;
            if (current.all <= p.all or current.task < p.task) break :blk 0;

            break :blk @intCast(@min(@as(u128, current.task - p.task) * 100 / (current.all - p.all), 100));
        };

        sum += chip.util[core];
    }

    return sum / @max(@as(u64, chip.usable.count()), 1);
}

/// Memory in use, summed over the groups of fused cores. Every core of a group reports the whole
/// group, so a group is read once, from its first core. A core not in use is a group of its own.
fn memoryInUse(self: *const Furiosa, chip: *Chip, in_use: Cores) ?u64 {
    const share = chip.model.memoryBytes() / cores;
    var used: u64 = 0;
    var left = chip.usable.count();
    var it = chip.usable.iterator(.{});

    while (it.next()) |head| {
        var size: usize = 1;
        var buffer: [1024]u8 = undefined;
        const text = if (in_use.isSet(head)) (self.coreFile(chip.node, head, "alloc_status", &buffer) catch return null) else null;

        if (text) |status| {
            const capacity = (hexField(status, "dram_capacity") orelse return null) + (hexField(status, "shared_capacity") orelse return null);
            const usage = (hexField(status, "dram_usage") orelse return null) + (hexField(status, "shared_usage") orelse return null);
            if (capacity > 0) {
                size = @intCast(@max(capacity / share, 1));
                chip.mem_used[head] = usage;
                used += usage;
            }
        }

        if (size > left) return null;
        left -= size;
        for (1..size) |_| _ = it.next();
    }

    return used;
}

fn coreFile(self: *const Furiosa, node: u8, core: usize, name: []const u8, buffer: []u8) !?[]const u8 {
    var path: [48]u8 = undefined;
    const text = self.class.read(try std.fmt.bufPrint(&path, "rngd!npu{d}pe{d}/{s}", .{ node, core, name }), buffer) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => |e| return e,
    };

    return if (text.len > 0) text else null;
}

/// Reads a milli-unit sysfs integer (millidegrees or microwatts) and scales it down by 1000.
fn milli(dir: smi.sysfs.Dir, file: ?[]const u8) ?u64 {
    const name = file orelse return null;
    const value = dir.int(u64, name) catch return null;

    return value / 1000;
}

fn coreClock(mgmt: smi.sysfs.Dir, usable: Cores) ?u64 {
    var buffer: [256]u8 = undefined;
    const text = mgmt.read("npu_clocks", &buffer) catch return null;

    var highest: ?u64 = null;
    var lines = std.mem.splitScalar(u8, text, '\n');
    var core: usize = 0;

    while (lines.next()) |line| : (core += 1) {
        const hz = std.fmt.parseInt(u64, std.mem.trim(u8, line, &std.ascii.whitespace), 10) catch return null;
        if (core < cores and usable.isSet(core)) {
            const mhz = hz / 1_000_000;
            highest = if (highest) |prev| @max(prev, mhz) else mhz;
        }
    }

    return highest;
}

fn occupancy(mgmt: smi.sysfs.Dir) ?Cores {
    var buffer: [64]u8 = undefined;
    const text = mgmt.read("pe_occupancy", &buffer) catch return null;

    var in_use: Cores = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    var core: usize = 0;

    while (lines.next()) |line| : (core += 1) {
        const state = std.mem.trim(u8, line, &std.ascii.whitespace);
        if (std.mem.eql(u8, state, "1")) {
            if (core < cores) in_use.set(core);
        } else if (!std.mem.eql(u8, state, "0")) return null;
    }

    return in_use;
}

fn governor(mgmt: smi.sysfs.Dir) ?smi.Sample.Furiosa.Governor {
    var buffer: [256]u8 = undefined;
    const text = mgmt.read("npu_governor", &buffer) catch return null;
    const line = std.mem.trimEnd(u8, std.mem.sliceTo(text, '\n'), &std.ascii.whitespace);

    for (std.enums.values(smi.Sample.Furiosa.Governor)) |profile| {
        if (std.mem.endsWith(u8, line, @tagName(profile))) return profile;
    }

    return null;
}

fn liveness(mgmt: smi.sysfs.Dir) ?smi.Sample.Furiosa.Liveness {
    var buffer: [64]u8 = undefined;
    const state = mgmt.read("device_state", &buffer) catch return null;

    return if (std.mem.eql(u8, state, "0")) .alive else .unknown;
}

fn throttleReasons(mgmt: smi.sysfs.Dir) ?u64 {
    var buffer: [128]u8 = undefined;
    const text = mgmt.read("throttle_reason", &buffer) catch return null;
    const start = if (std.mem.findLastAny(u8, text, &std.ascii.whitespace)) |space| space + 1 else 0;

    return std.fmt.parseInt(u64, text[start..], 0) catch null;
}

fn hexField(text: []const u8, key: []const u8) ?u64 {
    var lines = std.mem.tokenizeScalar(u8, text, '\n');

    while (lines.next()) |line| {
        if (smi.sysfs.fieldValue(std.mem.trim(u8, line, " \t"), key)) |value| return std.fmt.parseInt(u64, value, 0) catch null;
    }

    return null;
}

fn deviceOf(self: *const Furiosa, node: u8) ?u16 {
    for (self.chips, 0..) |chip, index| {
        if (chip.node == node) return @intCast(index);
    }

    return null;
}

/// A bare npu<N> is core 0, as furiosa-smi-go reads it. Other nodes, such as npu<N>ch0, name no core.
const Node = struct {
    chip: u8,
    cores: Cores,

    fn parse(name: []const u8) ?Node {
        const rest = std.mem.cutPrefix(u8, name, "npu") orelse return null;
        const end = std.mem.findNone(u8, rest, "0123456789") orelse rest.len;
        const chip = std.fmt.parseUnsigned(u8, rest[0..end], 10) catch return null;

        const range = if (end == rest.len) "0" else std.mem.cutPrefix(u8, rest[end..], "pe") orelse return null;
        const first, const last = std.mem.cutScalar(u8, range, '-') orelse .{ range, range };
        const start = std.fmt.parseUnsigned(u8, first, 10) catch return null;
        const stop = std.fmt.parseUnsigned(u8, last, 10) catch return null;
        if (start > stop or stop >= cores) return null;

        var set: Cores = .empty;
        set.setRangeValue(.{ .start = start, .end = @as(usize, stop) + 1 }, true);

        return .{ .chip = chip, .cores = set };
    }
};

fn mgmtNode(name: []const u8) ?u8 {
    const rest = std.mem.cutPrefix(u8, name, "rngd!npu") orelse return null;
    const digits = std.mem.cutSuffix(u8, rest, "mgmt") orelse return null;

    return std.fmt.parseUnsigned(u8, digits, 10) catch null;
}

test Node {
    try std.testing.expectEqual(Node{ .chip = 3, .cores = .{ .mask = 0x0f } }, Node.parse("npu3pe0-3").?);
    try std.testing.expectEqual(Node{ .chip = 0, .cores = .{ .mask = 0x20 } }, Node.parse("npu0pe5").?);
    try std.testing.expectEqual(Node{ .chip = 12, .cores = .{ .mask = 0x01 } }, Node.parse("npu12").?);

    for ([_][]const u8{ "npu0ch0", "npu0ch0r", "npu0mgmt", "npu0bar4", "npu0dmar", "npu0pe3-1", "npu0pe8", "npu0pe", "npu" }) |name| {
        try std.testing.expectEqual(null, Node.parse(name));
    }

    try std.testing.expectEqual(7, mgmtNode("rngd!npu7mgmt"));
    try std.testing.expectEqual(null, mgmtNode("rngd!npu7pe0"));
}
