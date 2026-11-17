//! FuriosaAI RNGD chips, through sysfs. Paths, units and formulas follow furiosa-smi-go 2026.4.0.
const std = @import("std");
const builtin = @import("builtin");
const contract = @import("zml-smi-2/contract");
const native = @import("zml-smi-2/native");
const source = @import("zml-smi-2/source");
const polling = @import("zml-smi-2/poll");

const Furiosa = @This();

devices: []const contract.Device,
/// /sys/class/rngd_mgmt: rngd!npu<N>mgmt for chip N, and rngd!npu<N>pe<C> for its core C.
class: native.sysfs.Dir,
chips: []Chip,

/// The cores (PEs) of a chip.
const cores = 8;
const Cores = std.bit_set.Integer(cores);

const Chip = struct {
    /// N in /dev/rngd/npu<N>.
    node: u8,
    model: *const Model,
    mgmt: native.sysfs.Dir,
    hwmon: ?Hwmon,
    /// The cores this host may use: those a node of /dev/rngd names.
    usable: Cores,
    cycles: [cores]Cycles = @splat(.{}),
    /// This round: the utilization of each core, and the memory in use of the group each core heads.
    util: [cores]u8 = @splat(0),
    mem_used: [cores]u64 = @splat(0),

    fn close(chip: Chip) void {
        chip.mgmt.close();
        if (chip.hwmon) |hwmon| hwmon.dir.close();
    }

    fn lessThan(_: void, first: Chip, second: Chip) bool {
        if (first.model.device != second.model.device) return first.model.device < second.model.device;
        return first.node < second.node;
    }

    /// The row of a process: the mean utilization of the cores it holds, and the memory their groups use.
    fn row(
        chip: *const Chip,
        pid: u32,
        device: u16,
        held: Cores,
        sample: *const contract.Metrics,
    ) contract.ProcessInfo {
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
            .dev_util_percent = if (sample.get(.util_percent) != null)
                @intCast(util / @max(@as(u64, mine.count()), 1))
            else
                null,
            .dev_mem_kib = if (sample.get(.mem_used_bytes) != null) mem_used / 1024 else null,
        };
    }
};

/// The running totals of a core: its cycles, and those it spent on tasks.
const Cycles = struct { all: native.Counter = .{}, task: native.Counter = .{} };

const Hwmon = struct {
    dir: native.sysfs.Dir,
    temperature: ?[]const u8,
    ambient: ?[]const u8,
    power: ?[]const u8,
    power_limit: ?[]const u8 = "power1_max",

    fn open(arena: std.mem.Allocator, device: native.sysfs.Dir) ?Hwmon {
        const dir = native.hwmon.find(device, null) catch return null;
        return .{
            .dir = dir,
            .temperature = native.hwmon.resolve(arena, dir, &.{.{ .prefix = "temp", .label = "PEAK" }}),
            .ambient = native.hwmon.resolve(arena, dir, &.{.{ .prefix = "temp", .label = "AMBIENT" }}),
            .power = native.hwmon.resolve(arena, dir, &.{.{ .prefix = "power", .label = "RMS_TOTAL", .suffix = "average" }}),
        };
    }
};

/// A device type. Its firmware reserves 512 MiB of its DRAM, and its cores share the rest evenly.
const Model = struct {
    device: u16,
    name: []const u8,
    memory_bytes: u64,
};

const models = [_]Model{
    .{ .device = 0x0001, .name = "RNGD", .memory_bytes = (48 << 30) - (512 << 20) },
    .{ .device = 0x0002, .name = "RNGD+", .memory_bytes = (72 << 30) - (512 << 20) },
};

const vendor = 0x1ed2;

pub const two_readings = true;

pub const detection_paths = [_][]const u8{"/dev/rngd"};

pub fn open(arena: std.mem.Allocator, _: std.mem.Allocator, io: std.Io) !Furiosa {
    const usable = try usableCores(io);
    const class = try native.sysfs.Dir.open(io, "/sys/class/rngd_mgmt");
    errdefer class.close();

    var chips: std.ArrayList(Chip) = .empty;
    errdefer for (chips.items) |chip| chip.close();
    var it = class.iterate();
    while (try it.next(io)) |entry| {
        const node = mgmtNode(entry.name) orelse continue;
        // A chip with no node in /dev/rngd is not this host's.
        if (usable[node].count() == 0) continue;
        const chip = openChip(arena, class, entry.name, node, usable[node]) catch continue;
        errdefer chip.close();
        try chips.append(arena, chip);
    }
    std.mem.sort(Chip, chips.items, {}, Chip.lessThan);

    const devices = try arena.alloc(contract.Device, chips.items.len);
    for (devices, chips.items) |*device, chip| {
        var details: std.ArrayList(contract.Device.Detail) = .empty;
        for ([_][]const u8{ "fw_version", "device_sn" }, [_][]const u8{
            "firmware_version", "serial",
        }) |file, key| {
            const value = string(arena, chip.mgmt, file) orelse continue;
            try details.append(arena, .{ .key = key, .value = value });
        }
        device.* = .{
            .target = .furiosa,
            .name = chip.model.name,
            .driver_version = string(arena, chip.mgmt, "version"),
            .details = details.items,
        };
    }

    return .{
        .devices = devices,
        .class = class,
        .chips = chips.items,
    };
}

/// The cores of each chip that the nodes of /dev/rngd name.
fn usableCores(io: std.Io) ![256]Cores {
    var usable: [256]Cores = @splat(.empty);
    const dev = try native.sysfs.Dir.open(io, "/dev/rngd");
    defer dev.close();
    var it = dev.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .character_device) continue;
        const node = Node.parse(entry.name) orelse continue;
        usable[node.chip].setUnion(node.cores);
    }
    return usable;
}

fn openChip(arena: std.mem.Allocator, class: native.sysfs.Dir, name: []const u8, node: u8, usable: Cores) !Chip {
    const mgmt = try class.openDir(name);
    errdefer mgmt.close();

    var buffer: [64]u8 = undefined;
    var path: [64]u8 = undefined;
    const pci = try native.sysfs.Dir.open(mgmt.io, try std.fmt.bufPrint(&path, "/sys/bus/pci/devices/{s}", .{try mgmt.read("busname", &buffer)}));
    defer pci.close();

    if (try pci.int(u16, "vendor") != vendor) return error.NotFuriosa;
    const device = try pci.int(u16, "device");
    const chip_model = for (&models) |*candidate| {
        if (candidate.device == device) break candidate;
    } else return error.UnknownDevice;

    return .{ .node = node, .model = chip_model, .mgmt = mgmt, .hwmon = .open(arena, pci), .usable = usable };
}

pub fn close(self: *Furiosa) void {
    for (self.chips) |chip| chip.close();
    self.class.close();
}

pub const poll = polling.PerDevice(@This()).poll;

pub const Probe = struct {
    backend: *const Furiosa,
    chip: *Chip,
    hwmon: ?*const Hwmon,

    pub const temperature = native.hwmon.Scalar(Probe, .{ .temperature, .milli, .one });
    pub const ambient_temperature = native.hwmon.Scalar(Probe, .{ .ambient, .milli, .one });
    pub const power_mw = native.hwmon.Scalar(Probe, .{ .power, .micro, .milli });
    pub const power_limit_mw = native.hwmon.Scalar(Probe, .{ .power_limit, .micro, .milli });

    const Workload = contract.Reply(&.{
        .util_percent, .mem_used_bytes, .mem_total_bytes,
    });

    pub fn workload(self: *const Probe) Workload {
        const chip = self.chip;
        chip.util = @splat(0);
        chip.mem_used = @splat(0);
        var reply: Workload = .{ .mem_total_bytes = chip.model.memory_bytes };
        if (occupancy(chip.mgmt)) |in_use| {
            reply.util_percent = self.backend.utilization(chip, in_use);
            reply.mem_used_bytes = self.backend.memoryInUse(chip, in_use);
        }
        return reply;
    }

    const Management = contract.Reply(&.{ .governor, .liveness, .throttle_reasons });

    pub fn management(self: *const Probe) Management {
        const dir = self.chip.mgmt;
        var buffer: [256]u8 = undefined;
        var reply: Management = .{};
        // The governor name ends the first line.
        if (dir.read("npu_governor", &buffer) catch null) |text| {
            const line = std.mem.trimEnd(u8, std.mem.sliceTo(text, '\n'), &std.ascii.whitespace);
            reply.governor = for (std.enums.values(contract.Governor)) |profile| {
                if (std.mem.endsWith(u8, line, @tagName(profile))) break @backingInt(profile);
            } else null;
        }
        if (dir.read("device_state", buffer[0..64]) catch null) |state| {
            const status: contract.Liveness = if (std.mem.eql(u8, state, "0")) .alive else .unknown;
            reply.liveness = @backingInt(status);
        }
        // The throttle bitmask ends the file, e.g. "Throttle reason = 0x00000000".
        if (dir.read("throttle_reason", buffer[0..128]) catch null) |text| {
            var words = std.mem.splitBackwardsAny(u8, text, &std.ascii.whitespace);
            reply.throttle_reasons = hex(words.first());
        }
        return reply;
    }
};

pub fn probe(self: *Furiosa, index: usize) Probe {
    const chip = &self.chips[index];
    return .{
        .backend = self,
        .chip = chip,
        .hwmon = if (chip.hwmon) |*hwmon| hwmon else null,
    };
}

pub fn readProcesses(self: *Furiosa, scratch: std.mem.Allocator, reading: *source.Reading) !void {
    // The cores each process holds on each chip, from the chip and core nodes it has open.
    var held: std.AutoArrayHashMapUnmanaged(struct { u32, u16 }, Cores) = .empty;
    defer held.deinit(scratch);
    var holders: native.proc.FdHolders = .init(self.class.io, &.{"/dev/rngd/"});
    defer holders.deinit();
    while (holders.next()) |holder| {
        const node = Node.parse(holder.rest) orelse continue;
        const device: u16 = for (self.chips, 0..) |chip, index| {
            if (chip.node == node.chip) break @intCast(index);
        } else continue;
        const entry = try held.getOrPutValue(scratch, .{ holder.pid, device }, .empty);
        entry.value_ptr.setUnion(node.cores);
    }
    for (held.keys(), held.values()) |key, set| {
        const pid, const device = key;
        const row = self.chips[device].row(pid, device, set, &reading.samples[device]);
        try reading.addProcess(row);
    }
}

/// The mean over the usable cores of the share of its cycles each spent on tasks.
fn utilization(self: *const Furiosa, chip: *Chip, in_use: Cores) ?u64 {
    var sum: u64 = 0;
    var it = chip.usable.iterator(.{});
    while (it.next()) |core| {
        var buffer: [1024]u8 = undefined;
        const text = if (in_use.isSet(core))
            self.coreFile(chip.node, core, "metrics", &buffer) catch return null
        else
            null;
        // A core not in use reads 0 cycles, as in furiosa-smi-go.
        const all = if (text) |metrics| (hexField(metrics, "CycleCount") orelse return null) else 0;
        const task = if (text) |metrics| (hexField(metrics, "TaskCycleCount") orelse return null) else 0;
        const all_growth = chip.cycles[core].all.advance(self.class.io, all);
        const task_growth = chip.cycles[core].task.advance(self.class.io, task);
        chip.util[core] = if (all_growth != null and task_growth != null) task_growth.?.percentOf(all_growth.?) else 0;
        sum += chip.util[core];
    }
    return sum / @max(@as(u64, chip.usable.count()), 1);
}

/// Memory in use, summed over the groups of fused cores. Every core of a group reports the whole
/// group, so a group is read once, from its first core. A core not in use is a group of its own.
fn memoryInUse(self: *const Furiosa, chip: *Chip, in_use: Cores) ?u64 {
    const share = chip.model.memory_bytes / cores;
    var used: u64 = 0;
    var it = chip.usable.iterator(.{});
    while (it.next()) |head| {
        var buffer: [1024]u8 = undefined;
        if (!in_use.isSet(head)) continue;
        const status = (self.coreFile(chip.node, head, "alloc_status", &buffer) catch return null) orelse
            continue;
        var totals: [2]u64 = undefined;
        inline for (.{ "capacity", "usage" }, &totals) |field, *total| {
            const dram = hexField(status, "dram_" ++ field) orelse return null;
            total.* = dram + (hexField(status, "shared_" ++ field) orelse return null);
        }
        const capacity, const usage = totals;
        if (capacity == 0) continue;
        chip.mem_used[head] = usage;
        used += usage;
        for (1..@max(capacity / share, 1)) |_| _ = it.next() orelse return null;
    }
    return used;
}

/// A file of a core, or null when the core is not in use: the file is then empty or missing.
fn coreFile(self: *const Furiosa, node: u8, core: usize, name: []const u8, buffer: []u8) !?[]const u8 {
    var path: [48]u8 = undefined;
    const file = try std.fmt.bufPrint(&path, "rngd!npu{d}pe{d}/{s}", .{ node, core, name });
    const text = self.class.read(file, buffer) catch |err| {
        return if (err == error.FileNotFound) null else err;
    };
    return if (text.len > 0) text else null;
}

/// The cores in use: one line per core, 1 when it is in use and 0 when it is not.
fn occupancy(mgmt: native.sysfs.Dir) ?Cores {
    var buffer: [64]u8 = undefined;
    const text = mgmt.read("pe_occupancy", &buffer) catch return null;
    var in_use: Cores = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    var core: usize = 0;
    while (lines.next()) |line| : (core += 1) {
        const state = std.mem.trim(u8, line, &std.ascii.whitespace);
        if (state.len != 1 or (state[0] != '0' and state[0] != '1')) return null;
        if (state[0] == '1' and core < cores) in_use.set(core);
    }
    return in_use;
}

/// The hex number of `key` in a file of "key: value" lines.
fn hexField(text: []const u8, key: []const u8) ?u64 {
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (native.sysfs.fieldValue(std.mem.trim(u8, line, " \t"), key)) |value| return hex(value);
    }
    return null;
}

/// A number in hex, with or without a 0x prefix.
fn hex(text: []const u8) ?u64 {
    const digits = std.mem.cutPrefix(u8, text, "0x") orelse std.mem.cutPrefix(u8, text, "0X") orelse text;
    return std.fmt.parseInt(u64, digits, 16) catch null;
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
fn string(arena: std.mem.Allocator, dir: native.sysfs.Dir, file: []const u8) ?[]const u8 {
    var buffer: [128]u8 = undefined;
    const text = dir.read(file, &buffer) catch return null;
    if (text.len == 0) return null;
    return arena.dupe(u8, text) catch null;
}

/// Private declarations exposed only to the sibling test module.
pub const testing = if (builtin.is_test) struct {
    pub const Node = Furiosa.Node;
    pub const Chip = Furiosa.Chip;
    pub const models = Furiosa.models;
    pub const cores = Furiosa.cores;
    pub const parseNode = Furiosa.Node.parse;
    pub const mgmtNode = Furiosa.mgmtNode;
    pub const processRow = Furiosa.Chip.row;
    pub const memoryInUse = Furiosa.memoryInUse;
} else struct {};
