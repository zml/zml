//! FuriosaAI RNGD chips, through sysfs. Paths, units and formulas follow furiosa-smi-go 2026.4.0.
const std = @import("std");
const smi = @import("zml-smi");

const Furiosa = @This();

platform: smi.Platform,
io: std.Io,
/// /sys/class/rngd_mgmt: rngd!npu<N>mgmt for chip N, and rngd!npu<N>pe<C> for its core C.
class: smi.sysfs.Dir,
chips: []Chip,

/// The cores (PEs) of a chip.
const cores = 8;
const Cores = std.bit_set.Integer(cores);

const Chip = struct {
    /// N in /dev/rngd/npu<N>.
    node: u8,
    model: *const Model,
    mgmt: smi.sysfs.Dir,
    hwmon: ?Hwmon,
    /// The cores this host may use: those a node of /dev/rngd names.
    usable: Cores,
    cycles: [cores]Cycles = @splat(.{}),
    /// This round: the utilization of each core, and the memory in use of the group each core heads.
    util: [cores]u8 = @splat(0),
    mem_used: [cores]u64 = @splat(0),

    fn lessThan(_: void, a: Chip, b: Chip) bool {
        if (a.model.device != b.model.device) return a.model.device < b.model.device;
        return a.node < b.node;
    }

    /// The row of a process: the mean utilization of the cores it holds, and the memory their groups use.
    fn row(chip: *const Chip, pid: u32, device: u16, held: Cores, sample: smi.Furiosa) smi.ProcessInfo {
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

/// The running totals of a core: its cycles, and those it spent on tasks.
const Cycles = struct { all: smi.Counter = .{}, task: smi.Counter = .{} };

const Hwmon = struct {
    dir: smi.sysfs.Dir,
    temperature: ?[]const u8,
    ambient: ?[]const u8,
    power: ?[]const u8,

    fn open(arena: std.mem.Allocator, device: smi.sysfs.Dir) ?Hwmon {
        const dir = smi.hwmon.find(device, null) catch return null;
        return .{
            .dir = dir,
            .temperature = smi.hwmon.resolve(arena, dir, &.{.{ .prefix = "temp", .label = "PEAK" }}),
            .ambient = smi.hwmon.resolve(arena, dir, &.{.{ .prefix = "temp", .label = "AMBIENT" }}),
            .power = smi.hwmon.resolve(arena, dir, &.{.{ .prefix = "power", .label = "RMS_TOTAL", .suffix = "average" }}),
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

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes, .deinit = deinit };

pub fn detect(io: std.Io) bool {
    return smi.sysfs.exists(io, "/dev/rngd");
}

pub fn init(arena: std.mem.Allocator, _: std.mem.Allocator, io: std.Io) !Furiosa {
    const usable = try usableCores(io);
    const class = try smi.sysfs.Dir.open(io, "/sys/class/rngd_mgmt");

    var chips: std.ArrayList(Chip) = .empty;
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
        try chips.append(arena, chip);
    }
    std.mem.sort(Chip, chips.items, {}, Chip.lessThan);

    const devices = try arena.alloc(smi.Device, chips.items.len);
    for (devices, chips.items) |*device, chip| {
        var details: std.ArrayList(smi.Device.Detail) = .empty;
        if (string(arena, chip.mgmt, "fw_version")) |value| try details.append(arena, .{ .key = "firmware_version", .label = "Firmware", .value = value });
        if (string(arena, chip.mgmt, "device_sn")) |value| try details.append(arena, .{ .key = "serial", .label = "Serial", .value = value });
        device.* = .{
            .target = .furiosa,
            .name = chip.model.name,
            .driver_version = string(arena, chip.mgmt, "version"),
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

/// The cores of each chip that the nodes of /dev/rngd name.
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
    defer pci.close();

    if (try pci.int(u16, "vendor") != vendor) return error.NotFuriosa;
    const device = try pci.int(u16, "device");
    const model = for (&models) |*model| {
        if (model.device == device) break model;
    } else return error.UnknownDevice;

    return .{ .node = node, .model = model, .mgmt = mgmt, .hwmon = .open(arena, pci), .usable = usable };
}

fn deinit(platform: *smi.Platform) void {
    const self: *Furiosa = @alignCast(@fieldParentPtr("platform", platform));
    for (self.chips) |chip| {
        chip.mgmt.close();
        if (chip.hwmon) |hwmon| hwmon.dir.close();
    }
    self.class.close();
}

fn poll(platform: *smi.Platform, _: std.mem.Allocator, samples: []smi.Sample) void {
    const self: *Furiosa = @alignCast(@fieldParentPtr("platform", platform));

    for (self.chips, samples) |*chip, *sample| {
        const fields = &sample.furiosa;
        fields.mem_total_bytes = chip.model.memoryBytes();
        fields.governor = governor(chip.mgmt);
        fields.liveness = liveness(chip.mgmt);
        fields.throttle_reasons = throttleReasons(chip.mgmt);

        chip.util = @splat(0);
        chip.mem_used = @splat(0);
        if (occupancy(chip.mgmt)) |in_use| {
            fields.util_percent = self.utilization(chip, in_use);
            fields.mem_used_bytes = self.memoryInUse(chip, in_use);
        }

        const hwmon = chip.hwmon orelse continue;
        // hwmon reports millidegrees and microwatts.
        if (hwmon.temperature) |file| fields.temperature = hwmon.dir.scaled(file, .milli, .one) catch null;
        if (hwmon.ambient) |file| fields.ambient_temperature = hwmon.dir.scaled(file, .milli, .one) catch null;
        if (hwmon.power) |file| fields.power_mw = hwmon.dir.scaled(file, .micro, .milli) catch null;
        fields.power_limit_mw = hwmon.dir.scaled("power1_max", .micro, .milli) catch null;
    }
}

/// The mean over the usable cores of the share of its cycles each spent on tasks.
fn utilization(self: *const Furiosa, chip: *Chip, in_use: Cores) ?u64 {
    var sum: u64 = 0;
    var it = chip.usable.iterator(.{});
    while (it.next()) |core| {
        var buffer: [1024]u8 = undefined;
        const text = if (in_use.isSet(core)) (self.coreFile(chip.node, core, "metrics", &buffer) catch return null) else null;
        // A core not in use reads 0 cycles, as in furiosa-smi-go.
        const all = if (text) |metrics| (hexField(metrics, "CycleCount") orelse return null) else 0;
        const task = if (text) |metrics| (hexField(metrics, "TaskCycleCount") orelse return null) else 0;
        const all_growth = chip.cycles[core].all.advance(self.io, all);
        const task_growth = chip.cycles[core].task.advance(self.io, task);
        chip.util[core] = if (all_growth != null and task_growth != null) task_growth.?.percentOf(all_growth.?) else 0;
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

/// A file of a core, or null when the core is not in use: the file is then empty or missing.
fn coreFile(self: *const Furiosa, node: u8, core: usize, name: []const u8, buffer: []u8) !?[]const u8 {
    var path: [48]u8 = undefined;
    const text = self.class.read(try std.fmt.bufPrint(&path, "rngd!npu{d}pe{d}/{s}", .{ node, core, name }), buffer) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => |e| return e,
    };
    return if (text.len > 0) text else null;
}

/// The cores in use: one line per core, 1 when it is in use and 0 when it is not.
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

/// The governor whose name ends the first line.
fn governor(mgmt: smi.sysfs.Dir) ?u64 {
    var buffer: [256]u8 = undefined;
    const text = mgmt.read("npu_governor", &buffer) catch return null;
    const line = std.mem.trimEnd(u8, std.mem.sliceTo(text, '\n'), &std.ascii.whitespace);
    for (std.enums.values(smi.Furiosa.Governor)) |profile| {
        if (std.mem.endsWith(u8, line, @tagName(profile))) return @backingInt(profile);
    }
    return null;
}

/// Alive when the device state is 0.
fn liveness(mgmt: smi.sysfs.Dir) ?u64 {
    var buffer: [64]u8 = undefined;
    const state = mgmt.read("device_state", &buffer) catch return null;
    return @backingInt(if (std.mem.eql(u8, state, "0")) smi.Furiosa.Liveness.alive else .unknown);
}

/// The bitmask that ends the file, in hex: "Throttle reason = 0x00000000".
fn throttleReasons(mgmt: smi.sysfs.Dir) ?u64 {
    var buffer: [128]u8 = undefined;
    const text = mgmt.read("throttle_reason", &buffer) catch return null;
    const start = if (std.mem.findLastAny(u8, text, &std.ascii.whitespace)) |space| space + 1 else 0;
    return hex(text[start..]);
}

/// The hex number of `key` in a file of "key: value" lines.
fn hexField(text: []const u8, key: []const u8) ?u64 {
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (smi.sysfs.fieldValue(std.mem.trim(u8, line, " \t"), key)) |value| return hex(value);
    }
    return null;
}

/// A number in hex, with or without a 0x prefix.
fn hex(text: []const u8) ?u64 {
    const digits = std.mem.cutPrefix(u8, text, "0x") orelse std.mem.cutPrefix(u8, text, "0X") orelse text;
    return std.fmt.parseInt(u64, digits, 16) catch null;
}

fn processes(platform: *smi.Platform, scratch: std.mem.Allocator, samples: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
    const self: *Furiosa = @alignCast(@fieldParentPtr("platform", platform));
    // The cores each process holds on each chip, from the chip and core nodes it has open.
    var held: std.AutoArrayHashMapUnmanaged(struct { u32, u16 }, Cores) = .empty;
    var holders: smi.proc.FdHolders = .init(self.io, &.{"/dev/rngd/"});
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

fn deviceOf(self: *const Furiosa, node: u8) ?u16 {
    for (self.chips, 0..) |chip, index| {
        if (chip.node == node) return @intCast(index);
    }
    return null;
}

/// The chip and cores a node of /dev/rngd names: npu<N>, npu<N>pe<C> or npu<N>pe<C>-<D>.
/// A bare npu<N> is core 0, as furiosa-smi-go reads it. Other nodes, such as npu<N>ch0, name no core.
const Node = struct {
    chip: u8,
    cores: Cores,

    fn parse(name: []const u8) ?Node {
        const rest = std.mem.cutPrefix(u8, name, "npu") orelse return null;
        const end = std.mem.findNone(u8, rest, "0123456789") orelse rest.len;
        const chip = decimal(rest[0..end]) orelse return null;
        const range = if (end == rest.len) "0" else std.mem.cutPrefix(u8, rest[end..], "pe") orelse return null;
        const first, const last = std.mem.cutScalar(u8, range, '-') orelse .{ range, range };
        const start = decimal(first) orelse return null;
        const stop = decimal(last) orelse return null;
        if (start > stop or stop >= cores) return null;
        var set: Cores = .empty;
        set.setRangeValue(.{ .start = start, .end = @as(usize, stop) + 1 }, true);
        return .{ .chip = chip, .cores = set };
    }
};

/// N in a management node name, rngd!npu<N>mgmt.
fn mgmtNode(name: []const u8) ?u8 {
    const rest = std.mem.cutPrefix(u8, name, "rngd!npu") orelse return null;
    return decimal(std.mem.cutSuffix(u8, rest, "mgmt") orelse return null);
}

fn decimal(text: []const u8) ?u8 {
    if (std.mem.findNone(u8, text, "0123456789") != null) return null;
    return std.fmt.parseInt(u8, text, 10) catch null;
}

/// A string of a sysfs file, copied into `arena`, or null when it is missing or empty.
fn string(arena: std.mem.Allocator, dir: smi.sysfs.Dir, file: []const u8) ?[]const u8 {
    var buffer: [128]u8 = undefined;
    const text = dir.read(file, &buffer) catch return null;
    if (text.len == 0) return null;
    return arena.dupe(u8, text) catch null;
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

test "utilization and memory of fused cores" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const gib: u64 = 1 << 30;
    const share = models[0].memoryBytes() / cores;

    // Cores 0 to 3 are one group, core 5 is alone, and core 6 was released after the occupancy was read:
    // the files of a core not in use are empty.
    try tmp.dir.createDirPath(io, "rngd!npu0mgmt");
    try tmp.dir.createDirPath(io, "rngd!npu0pe6");
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0pe6/metrics", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0pe6/alloc_status", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0mgmt/pe_occupancy", .data = "1\n1\n1\n1\n0\n1\n1\n0\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0mgmt/npu_governor", .data = "powersave\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0mgmt/device_state", .data = "0\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0mgmt/throttle_reason", .data = "Throttle reason = 0x00000003\n" });
    var text: [128]u8 = undefined;
    var path: [32]u8 = undefined;
    for ([_]u8{ 0, 1, 2, 3, 5 }) |core| {
        const fused = core < 4;
        try tmp.dir.createDirPath(io, try std.fmt.bufPrint(&path, "rngd!npu0pe{d}", .{core}));
        try tmp.dir.writeFile(io, .{
            .sub_path = try std.fmt.bufPrint(&path, "rngd!npu0pe{d}/alloc_status", .{core}),
            .data = try std.fmt.bufPrint(&text, "dram_capacity: 0x{x}\ndram_usage: 0x{x}\nshared_capacity: 0x0\nshared_usage: 0x0\n", .{
                if (fused) 4 * share else share, if (fused) 3 * gib else gib,
            }),
        });
    }

    const class: smi.sysfs.Dir = .{ .handle = tmp.dir, .io = io };
    var chips = [_]Chip{.{ .node = 0, .model = &models[0], .mgmt = try class.openDir("rngd!npu0mgmt"), .hwmon = null, .usable = .full }};
    defer chips[0].mgmt.close();
    var furiosa: Furiosa = .{ .platform = .{ .devices = &.{.{ .target = .furiosa }}, .vtable = &vtable }, .io = io, .class = class, .chips = &chips };

    // Between the polls, the group spends half its cycles on tasks and core 5 all of them.
    for ([_]u64{ 0, 37 }, [_]u64{ 1000, 2000 }) |util, cycles| {
        for ([_]u8{ 0, 1, 2, 3, 5 }) |core| {
            const task = if (core == 5) cycles else cycles / 2 + 250;
            try tmp.dir.writeFile(io, .{
                .sub_path = try std.fmt.bufPrint(&path, "rngd!npu0pe{d}/metrics", .{core}),
                .data = try std.fmt.bufPrint(&text, "CycleCount: 0x{x}\nTaskCycleCount: 0x{x}\n", .{ cycles, task }),
            });
        }
        var samples = [_]smi.Sample{.init(.furiosa)};
        poll(&furiosa.platform, std.testing.allocator, &samples);
        const chip = samples[0].furiosa;
        try std.testing.expectEqual(util, chip.util_percent);
        try std.testing.expectEqual(4 * gib, chip.mem_used_bytes);
        try std.testing.expectEqual(@backingInt(smi.Furiosa.Governor.powersave), chip.governor);
        try std.testing.expectEqual(@backingInt(smi.Furiosa.Liveness.alive), chip.liveness);
        try std.testing.expectEqual(0b11, chip.throttle_reasons);
    }

    // A process that holds the group sees its share.
    const row = chips[0].row(7, 0, .{ .mask = 0x0f }, .{ .util_percent = 37, .mem_used_bytes = 4 * gib });
    try std.testing.expectEqual(50, row.dev_util_percent);
    try std.testing.expectEqual(3 * gib / 1024, row.dev_mem_kib);
}
