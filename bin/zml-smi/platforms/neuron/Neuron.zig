//! AWS Neuron cores, through the bundled libnrt and sysfs.
const std = @import("std");
const c = @import("c");
const model = @import("zml-smi/model");
const source = @import("zml-smi/source");
const native = @import("zml-smi/native");
const telemetry = @import("zml-smi/telemetry");

const Neuron = @This();
const log = std.log.scoped(.@"zml-smi/neuron");

devices: []const model.Device,
io: std.Io,
nrt: Private,
chips: []const Chip,
/// Busy nanoseconds of each process on each core.
busy: native.Counters(Key),

/// A Neuron device. Each of its cores is a device of the platform.
const Chip = struct {
    handle: *c.ndl_device_t,
    mem_total_bytes: ?u64,
    /// stats/memory_usage/device_mem of each core.
    memory: []const ?native.sysfs.Dir,
};

/// A process and the position of a core in `devices`.
const Key = struct { pid: u32, core: u16 };

pub const two_readings = true;

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

const Checked = native.dynlib.CheckedBy(check);

/// The libnrt functions that report a status, exported or not, return 0 on success.
fn check(status: i64) error{NrtError}!void {
    if (status != 0) return error.NrtError;
}

pub fn detect(io: std.Io) bool {
    return native.sysfs.exists(io, c.NEURON_DEVICE_PREFIX ++ "0");
}

pub fn open(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !Neuron {
    const api = try native.dynlib.openBundled(Api, io, "libnrt.so.1");
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
    const devices = try arena.alloc(model.Device, chips.len * cores_per_chip);
    const memory = try arena.alloc(?native.sysfs.Dir, devices.len);
    @memset(memory, null);
    errdefer closeMemory(memory);

    for (chips, opened.items, 0..) |*chip, entry, index| {
        const device_index, const handle = entry;
        const first = index * cores_per_chip;
        const cores = memory[first..][0..cores_per_chip];
        const name = try openMemory(arena, io, device_index, cores);
        chip.* = .{ .handle = handle, .mem_total_bytes = memoryTotal(handle.device_type), .memory = cores };
        const device: model.Device = .{ .target = .neuron, .name = name, .driver_version = driver_version };
        @memset(devices[first..][0..cores_per_chip], device);
    }

    return .{
        .devices = devices,
        .io = io,
        .nrt = nrt,
        .chips = chips,
        .busy = .init(gpa),
    };
}

/// Opens the memory directory of each core and reads the chip's name.
fn openMemory(
    arena: std.mem.Allocator,
    io: std.Io,
    index: c_int,
    memory: []?native.sysfs.Dir,
) !?[]const u8 {
    var path: [64]u8 = undefined;
    const device_path = try std.fmt.bufPrint(&path, "/sys/devices/virtual/neuron_device/neuron{d}", .{index});
    const dir = native.sysfs.Dir.open(io, device_path) catch return null;
    defer dir.close();
    for (memory, 0..) |*core, core_index| {
        core.* = dir.openDir(try std.fmt.bufPrint(&path, "neuron_core{d}/stats/memory_usage/device_mem", .{core_index})) catch null;
    }
    var buffer: [128]u8 = undefined;
    return try arena.dupe(u8, dir.read("info/architecture/device_name", &buffer) catch return null);
}

/// Device memory by device type: Inferentia, then Inferentia2 and Trainium, Trainium2, Trainium3.
fn memoryTotal(device_type: u8) ?u64 {
    const gib: u64 = 1 << 30;
    const totals = [_]?u64{ null, 8 * gib, 32 * gib, 96 * gib, 144 * gib };
    return if (device_type < totals.len) totals[device_type] else null;
}

fn closeMemory(memory: []const ?native.sysfs.Dir) void {
    for (memory) |entry| if (entry) |dir| dir.close();
}

pub fn close(self: *Neuron) void {
    for (self.chips) |chip| closeMemory(chip.memory);
    self.busy.deinit();
}

const pollMemory = telemetry.Bind(model.Neuron, .{
    .read = readCategory,
    .fields = .{
        .nc_tensors = "tensors",
        .nc_constants = "constants",
        .nc_model_code = "model_code",
        .nc_shared_scratchpad = "model_shared_scratchpad",
        .nc_nonshared_scratchpad = "nonshared_scratchpad",
        .nc_runtime = "runtime_memory",
        .nc_driver = "driver_memory",
        .nc_dma_rings = "dma_rings",
        .nc_collectives = "collectives",
        .nc_notifications = "notifications",
        .nc_uncategorized = "uncategorized",
    },
}).poll;

fn readCategory(dir: native.sysfs.Dir, comptime category: []const u8) ?u64 {
    return dir.int(u64, category ++ "/total") catch null;
}

pub fn poll(self: *Neuron, scratch: std.mem.Allocator, reading: *source.Reading) !void {
    const samples = reading.samples;
    defer self.busy.sweep();
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
            if (memory) |dir| pollMemory(core, .{dir});
        }

        if (self.scan(scratch, chip.handle, @intCast(first), cores, reading)) |_| {
            read_any = true;
        } else |err| if (err == error.OutOfMemory) return err;
    }
    if (!read_any) return error.Unavailable;
    for (reading.processes.items) |*row| {
        row.dev_util_percent = @intCast(samples[row.device_idx].neuron.util_percent.?);
    }
}

fn memoryInUse(memory: []const ?native.sysfs.Dir) ?u64 {
    var total: u64 = 0;
    for (memory) |dir| total += (dir orelse return null).int(u64, "total") catch return null;
    return total;
}

/// Reads the busy time of every process on one chip: the utilization of each core,
/// and a row per process and core it has used, if it holds device memory.
fn scan(
    self: *Neuron,
    scratch: std.mem.Allocator,
    handle: *c.ndl_device_t,
    first: u16,
    cores: []model.Sample,
    reading: *source.Reading,
) !void {
    var list: ?[*]c.struct_neuron_app_info = null;
    var count: usize = 0;
    try self.nrt.ndl_get_all_apps_info.call(.{ handle, &list, &count, c.APP_INFO_ALL });
    defer std.c.free(list);
    const apps: []const c.struct_neuron_app_info = if (list) |items| items[0..count] else &.{};

    const busy = try scratch.alloc(?native.Growth, cores.len);
    @memset(busy, null);
    for (apps) |app| {
        if (app.pid <= 0) continue;
        const nds = (self.nrt.nds_open.get(.{ handle, app.pid }) catch continue) orelse continue;
        defer _ = self.nrt.nds_close(nds);

        for (busy, 0..) |*sum, core| {
            const micros = self.nrt.nds_get_nc_counter.get(.{ nds, @intCast(core), c.NDS_NC_COUNTER_TIME_IN_USE }) catch continue;
            const key: Key = .{ .pid = @intCast(app.pid), .core = first + @as(u16, @intCast(core)) };
            const nanos = native.units.scale(micros, .micro, .nano);
            if (try self.busy.advance(key, self.io, nanos)) |growth| native.Growth.add(sum, growth);

            const mem_kib = app.device_mem_size / 1024;
            if (micros > 0 and mem_kib > 0) try reading.addProcess(.{ .pid = key.pid, .device_idx = key.core, .dev_mem_kib = mem_kib });
        }
    }
    for (cores, busy) |*sample, sum| {
        sample.neuron.util_percent = if (sum) |growth| growth.percentOfTime() else 0;
    }
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
    inline for (info.field_names, info.field_types) |field, Field| {
        const address = symbol_blk: for (sections) |section| {
            if (section.type != .SYMTAB) continue;
            const strings = image[sections[section.link].offset..][0..sections[section.link].size];
            const symbols = std.mem.bytesAsSlice(elf.Sym, image[section.offset..][0..section.size]);
            for (symbols) |symbol| {
                if (symbol.value == 0) continue;
                if (std.mem.eql(u8, std.mem.sliceTo(strings[symbol.name..], 0), field)) {
                    break :symbol_blk @intFromPtr(library.fbase) + symbol.value;
                }
            }
        } else {
            log.warn("libnrt has no symbol {s}", .{field});
            return error.MissingSymbol;
        };
        @field(private, field) = if (@typeInfo(Field) == .@"struct")
            .{ .function = @ptrFromInt(address) }
        else
            @ptrFromInt(address);
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

pub const testing = if (@import("builtin").is_test) struct {
    pub const Api = Neuron.Api;
    pub const findPrivate = Neuron.findPrivate;
} else struct {};
