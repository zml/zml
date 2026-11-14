//! AWS Neuron cores, through the bundled libnrt and sysfs.
const std = @import("std");
const c = @import("c");
const smi = @import("zml-smi");

const Neuron = @This();
const log = std.log.scoped(.@"zml-smi/neuron");

platform: smi.Platform,
io: std.Io,
nrt: Private,
chips: []const Chip,
/// Busy nanoseconds of each process on each core.
busy: smi.Counters(Key),
/// The rows of this round's scan, in scratch. Null when no chip could be read.
rows: ?[]const smi.ProcessInfo = null,

/// A Neuron device. Each of its cores is a device of the platform.
const Chip = struct {
    handle: *c.ndl_device_t,
    name: ?[]const u8 = null,
    mem_total_bytes: ?u64,
    /// stats/memory_usage/device_mem of each core.
    memory: []const ?smi.sysfs.Dir,
};

/// A process and the position of a core in `platform.devices`.
const Key = struct { pid: u32, core: u16 };

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes, .deinit = deinit };

const Api = struct {
    nrt_get_total_nc_count: Checked(c.nrt_get_total_nc_count),
    nrt_get_version: Checked(c.nrt_get_version),
};

/// The functions libnrt does not export, found in its symbol table.
const Private = struct {
    ndl_available_devices: *const @TypeOf(c.ndl_available_devices),
    ndl_open_device: Checked(c.ndl_open_device),
    ndl_get_all_apps_info: Checked(c.ndl_get_all_apps_info),
    nds_open: Checked(c.nds_open),
    nds_close: *const @TypeOf(c.nds_close),
    nds_get_nc_counter: Checked(c.nds_get_nc_counter),
};

fn Checked(comptime function: anytype) type {
    return smi.dynlib.Checked(function, check);
}

/// The libnrt functions that report a status, exported or not, return 0 on success.
fn check(status: i64) error{NrtError}!void {
    if (status != 0) return error.NrtError;
}

pub fn detect(io: std.Io) bool {
    return smi.sysfs.exists(io, c.NEURON_DEVICE_PREFIX ++ "0");
}

pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !Neuron {
    const api = try smi.dynlib.openBundled(Api, io, "libnrt.so.1");
    const nrt = try findPrivate(io, api.nrt_get_version.function);
    const core_count = try api.nrt_get_total_nc_count.get(.{});

    var version = std.mem.zeroes(c.nrt_version_t);
    const driver_version = if (api.nrt_get_version.call(.{ &version, @sizeOf(c.nrt_version_t) })) |_|
        try arena.dupe(u8, std.mem.sliceTo(&version.rt_detail, 0))
    else |_|
        null;

    var indices: [c.MAX_NEURON_DEVICE_COUNT]c_int = undefined;
    const count: usize = @intCast(std.math.clamp(nrt.ndl_available_devices(&indices, indices.len), 0, indices.len));
    var opened: std.ArrayList(struct { c_int, *c.ndl_device_t }) = try .initCapacity(arena, count);
    for (indices[0..count]) |index| {
        var params: c.ndl_device_init_param_t = .{ .initialize_device = false, .num_dram_regions = 0, .map_hbm = false, .skip_copy_bufs = true };
        // A device that cannot be opened is skipped.
        const handle: ?*c.ndl_device_t = nrt.ndl_open_device.get(.{ index, &params }) catch continue;
        opened.appendAssumeCapacity(.{ index, handle orelse continue });
    }

    // The devices share the cores evenly.
    const cores_per_chip = core_count / @max(1, opened.items.len);
    const chips = try arena.alloc(Chip, opened.items.len);
    var devices: std.ArrayList(smi.Device) = try .initCapacity(arena, chips.len * cores_per_chip);
    for (chips, opened.items) |*chip, entry| {
        const index, const handle = entry;
        chip.* = try openChip(arena, io, index, handle, cores_per_chip);
        for (chip.memory) |_| devices.appendAssumeCapacity(.{ .target = .neuron, .name = chip.name, .driver_version = driver_version });
    }

    return .{
        .platform = .{ .devices = devices.items, .two_readings = true, .vtable = &vtable },
        .io = io,
        .nrt = nrt,
        .chips = chips,
        .busy = .init(gpa),
    };
}

/// A chip's memory total from its device type, and its name and the memory directory of each core from sysfs.
fn openChip(arena: std.mem.Allocator, io: std.Io, index: c_int, handle: *c.ndl_device_t, cores: usize) !Chip {
    const memory = try arena.alloc(?smi.sysfs.Dir, cores);
    @memset(memory, null);
    var chip: Chip = .{ .handle = handle, .mem_total_bytes = memoryTotal(handle.device_type), .memory = memory };

    var path: [64]u8 = undefined;
    const dir = smi.sysfs.Dir.open(io, try std.fmt.bufPrint(&path, "/sys/devices/virtual/neuron_device/neuron{d}", .{index})) catch return chip;
    defer dir.close();
    for (memory, 0..) |*core, core_index| {
        core.* = dir.openDir(try std.fmt.bufPrint(&path, "neuron_core{d}/stats/memory_usage/device_mem", .{core_index})) catch null;
    }
    var buffer: [128]u8 = undefined;
    chip.name = try arena.dupe(u8, dir.read("info/architecture/device_name", &buffer) catch return chip);
    return chip;
}

/// Device memory by device type: Inferentia, then Inferentia2 and Trainium, Trainium2, Trainium3.
fn memoryTotal(device_type: u8) ?u64 {
    const gib: u64 = 1 << 30;
    return switch (device_type) {
        1 => 8 * gib,
        2 => 32 * gib,
        3 => 96 * gib,
        4 => 144 * gib,
        else => null,
    };
}

fn deinit(platform: *smi.Platform) void {
    const self: *Neuron = @alignCast(@fieldParentPtr("platform", platform));
    for (self.chips) |chip| {
        for (chip.memory) |memory| if (memory) |dir| dir.close();
    }
    self.busy.deinit();
}

fn poll(platform: *smi.Platform, scratch: std.mem.Allocator, samples: []smi.Sample) void {
    const self: *Neuron = @alignCast(@fieldParentPtr("platform", platform));
    defer self.busy.sweep();
    var rows: std.ArrayList(smi.ProcessInfo) = .empty;
    var read_any = false;

    var first: usize = 0;
    for (self.chips) |chip| {
        const cores = samples[first..][0..chip.memory.len];
        defer first += cores.len;

        // The cores of a chip share its memory: each shows what all of them use.
        const mem_used = memoryInUse(chip.memory);
        for (cores, chip.memory) |*sample, memory| {
            const core = &sample.neuron;
            core.mem_used_bytes = mem_used;
            core.mem_total_bytes = chip.mem_total_bytes;
            const dir = memory orelse continue;
            core.nc_tensors = dir.int(u64, "tensors/total") catch null;
            core.nc_constants = dir.int(u64, "constants/total") catch null;
            core.nc_model_code = dir.int(u64, "model_code/total") catch null;
            core.nc_shared_scratchpad = dir.int(u64, "model_shared_scratchpad/total") catch null;
            core.nc_nonshared_scratchpad = dir.int(u64, "nonshared_scratchpad/total") catch null;
            core.nc_runtime = dir.int(u64, "runtime_memory/total") catch null;
            core.nc_driver = dir.int(u64, "driver_memory/total") catch null;
            core.nc_dma_rings = dir.int(u64, "dma_rings/total") catch null;
            core.nc_collectives = dir.int(u64, "collectives/total") catch null;
            core.nc_notifications = dir.int(u64, "notifications/total") catch null;
            core.nc_uncategorized = dir.int(u64, "uncategorized/total") catch null;
        }

        self.scan(scratch, chip.handle, @intCast(first), cores, &rows) catch continue;
        read_any = true;
    }
    self.rows = if (read_any) rows.items else null;
}

fn processes(platform: *smi.Platform, _: std.mem.Allocator, samples: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
    const self: *Neuron = @alignCast(@fieldParentPtr("platform", platform));
    for (self.rows orelse return error.Unavailable) |row| {
        var copy = row;
        if (samples[row.device_idx].neuron.util_percent) |util| copy.dev_util_percent = @intCast(util);
        try out.append(copy);
    }
}

fn memoryInUse(memory: []const ?smi.sysfs.Dir) ?u64 {
    var total: u64 = 0;
    for (memory) |dir| total += (dir orelse return null).int(u64, "total") catch return null;
    return total;
}

/// Reads the busy time of every process on one chip: the utilization of each core,
/// and a row per process and core it has used, if it holds device memory.
fn scan(self: *Neuron, scratch: std.mem.Allocator, handle: *c.ndl_device_t, first: u16, cores: []smi.Sample, rows: *std.ArrayList(smi.ProcessInfo)) !void {
    var list: ?[*]c.struct_neuron_app_info = null;
    var count: usize = 0;
    try self.nrt.ndl_get_all_apps_info.call(.{ handle, &list, &count, c.APP_INFO_ALL });
    defer std.c.free(list);
    const apps: []const c.struct_neuron_app_info = if (list) |items| items[0..count] else &.{};

    const busy = try scratch.alloc(?smi.Growth, cores.len);
    @memset(busy, null);
    for (apps) |app| {
        if (app.pid <= 0) continue;
        const nds = (self.nrt.nds_open.get(.{ handle, app.pid }) catch continue) orelse continue;
        defer _ = self.nrt.nds_close(nds);

        for (busy, 0..) |*sum, core| {
            const micros = self.nrt.nds_get_nc_counter.get(.{ nds, @intCast(core), c.NDS_NC_COUNTER_TIME_IN_USE }) catch continue;
            const key: Key = .{ .pid = @intCast(app.pid), .core = first + @as(u16, @intCast(core)) };
            if (try self.busy.advance(key, self.io, smi.units.scale(micros, .micro, .nano))) |growth| smi.Growth.add(sum, growth);

            const mem_kib = app.device_mem_size / 1024;
            if (micros > 0 and mem_kib > 0) try rows.append(scratch, .{ .pid = key.pid, .device_idx = key.core, .dev_mem_kib = mem_kib });
        }
    }
    for (cores, busy) |*sample, sum| sample.neuron.util_percent = if (sum) |growth| growth.percentOfTime() else 0;
}

/// Finds the functions of `Private` in the symbol table of the library that holds `known`.
fn findPrivate(io: std.Io, known: *const anyopaque) !Private {
    var library: DlInfo = undefined;
    if (dladdr(known, &library) == 0) return error.LibraryNotFound;
    const file = try std.Io.Dir.cwd().openFile(io, std.mem.span(library.fname), .{});
    defer file.close(io);
    var map = try file.createMemoryMap(io, .{ .len = @intCast((try file.stat(io)).size), .protection = .{ .read = true }, .populate = false });
    defer map.destroy(io);

    const elf = std.elf.ElfN;
    const image = map.memory;
    const header = std.mem.bytesAsValue(elf.Ehdr, image[0..@sizeOf(elf.Ehdr)]);
    const sections = std.mem.bytesAsSlice(elf.Shdr, image[header.shoff..][0 .. @as(usize, header.shnum) * @sizeOf(elf.Shdr)]);

    const info = @typeInfo(Private).@"struct";
    var private: Private = undefined;
    var found: [info.field_names.len]bool = @splat(false);
    for (sections) |section| {
        if (section.type != .SYMTAB) continue;
        const strings = image[sections[section.link].offset..][0..sections[section.link].size];
        for (std.mem.bytesAsSlice(elf.Sym, image[section.offset..][0..section.size])) |symbol| {
            if (symbol.value == 0) continue;
            const name = std.mem.sliceTo(strings[symbol.name..], 0);
            inline for (info.field_names, info.field_types, &found) |field, Field, *seen| {
                if (!seen.* and std.mem.eql(u8, name, field)) {
                    seen.* = true;
                    const address = @intFromPtr(library.fbase) + symbol.value;
                    @field(private, field) = if (@typeInfo(Field) == .@"struct") .{ .function = @ptrFromInt(address) } else @ptrFromInt(address);
                }
            }
        }
    }
    inline for (info.field_names, found) |field, seen| {
        if (!seen) {
            log.warn("libnrt has no symbol {s}", .{field});
            return error.MissingSymbol;
        }
    }
    return private;
}

const DlInfo = extern struct {
    fname: [*:0]const u8,
    fbase: *anyopaque,
    sname: ?[*:0]const u8,
    saddr: ?*anyopaque,
};

extern fn dladdr(address: *const anyopaque, info: *DlInfo) c_int;

test findPrivate {
    const io = std.testing.io;
    const api = try smi.dynlib.openBundled(Api, io, "libnrt.so.1");
    const nrt = try findPrivate(io, api.nrt_get_version.function);
    // Listing devices works without any, so the call checks the address.
    var indices: [c.MAX_NEURON_DEVICE_COUNT]c_int = undefined;
    try std.testing.expect(nrt.ndl_available_devices(&indices, indices.len) >= 0);
}

test "utilization and process rows" {
    const fake = struct {
        var busy_us: u64 = 1000;

        fn apps(device: [*c]c.ndl_device_t, info: [*c][*c]c.struct_neuron_app_info, count: [*c]usize, _: u16) callconv(.c) c_int {
            if (@intFromPtr(device) == 0x1000) return -1;
            const list: [*]c.struct_neuron_app_info = @ptrCast(@alignCast(std.c.malloc(3 * @sizeOf(c.struct_neuron_app_info)).?));
            // pid 0 is skipped, and pid 8 holds less than 1 KiB.
            for (list[0..3], [_]i32{ 0, 7, 8 }, [_]usize{ 1 << 20, 1 << 20, 512 }) |*app, pid, bytes| {
                app.* = std.mem.zeroes(c.struct_neuron_app_info);
                app.pid = pid;
                app.device_mem_size = bytes;
            }
            info.* = list;
            count.* = 3;
            return 0;
        }

        fn open(_: [*c]c.ndl_device_t, pid: c.pid_t, instance: [*c]?*c.nds_instance_t) callconv(.c) c_int {
            instance.* = @ptrFromInt(@as(usize, @intCast(pid)));
            return 0;
        }

        fn close(_: ?*c.nds_instance_t) callconv(.c) c_int {
            return 0;
        }

        /// Every process is busy on the first core of a chip.
        fn counter(_: ?*c.nds_instance_t, core: c_int, _: u32, value: [*c]u64) callconv(.c) c_int {
            value.* = if (core == 0) busy_us else 0;
            return 0;
        }
    };

    const devices: [4]smi.Device = @splat(.{ .target = .neuron });
    var neuron: Neuron = .{
        .platform = .{ .devices = &devices, .vtable = &vtable },
        .io = std.testing.io,
        .nrt = .{
            .ndl_available_devices = undefined,
            .ndl_open_device = undefined,
            .ndl_get_all_apps_info = .{ .function = &fake.apps },
            .nds_open = .{ .function = &fake.open },
            .nds_close = &fake.close,
            .nds_get_nc_counter = .{ .function = &fake.counter },
        },
        // The first chip cannot be read.
        .chips = &.{
            .{ .handle = @ptrFromInt(0x1000), .mem_total_bytes = null, .memory = &.{ null, null } },
            .{ .handle = @ptrFromInt(0x2000), .mem_total_bytes = null, .memory = &.{ null, null } },
        },
        .busy = .init(std.testing.allocator),
    };
    defer neuron.busy.deinit();
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    var list: smi.ProcessList = .init(std.testing.allocator);
    defer list.deinit();

    // The first reading has no growth; the second grows by far more than the time between them.
    for ([_]u8{ 0, 100 }) |util| {
        var samples: [4]smi.Sample = @splat(.init(.neuron));
        list.clear();
        poll(&neuron.platform, scratch.allocator(), &samples);
        try processes(&neuron.platform, scratch.allocator(), &samples, &list);

        try std.testing.expectEqual(null, samples[0].neuron.util_percent);
        try std.testing.expectEqual(util, samples[2].neuron.util_percent);
        try std.testing.expectEqual(0, samples[3].neuron.util_percent);
        const row: smi.ProcessInfo = .{ .pid = 7, .device_idx = 2, .dev_util_percent = util, .dev_mem_kib = 1024 };
        try std.testing.expectEqualDeep(@as([]const smi.ProcessInfo, &.{row}), list.rows.items);
        fake.busy_us += 1_000_000_000;
    }
}
