//! AWS Neuron cores, through the bundled libnrt and sysfs.
const std = @import("std");
const c = @import("c");
const contract = @import("zml-smi-2/contract");
const source = @import("zml-smi-2/source");
const native = @import("zml-smi-2/native");
const polling = @import("zml-smi-2/poll");

const Dir = native.sysfs.Dir;
const Neuron = @This();
const log = std.log.scoped(.@"zml-smi-2/neuron");

devices: []const contract.Device,
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
    memory: []const ?Dir,
};

/// A process and the position of a core in `devices`.
const Key = struct { pid: u32, core: u16 };

pub const two_readings = true;

const Api = native.dynlib.Bindings(c, check, &.{ "nrt_get_total_nc_count", "nrt_get_version" });

/// The functions libnrt does not export, found in its symbol table.
const Private = native.dynlib.Bindings(c, check, &.{
    "ndl_available_devices",
    "ndl_open_device",
    "ndl_get_all_apps_info",
    "nds_open",
    "nds_close",
    "nds_get_nc_counter",
});

/// The libnrt functions that report a status, exported or not, return 0 on success.
fn check(status: i64) error{NrtError}!void {
    if (status != 0) return error.NrtError;
}

pub const detection_paths = [_][]const u8{c.NEURON_DEVICE_PREFIX ++ "0"};

pub fn open(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !Neuron {
    const api = try native.dynlib.openBundled(Api, io, "libnrt.so.1");
    const nrt = try findPrivate(io, api.nrt_get_version.function);
    const core_count = try api.nrt_get_total_nc_count.get(.{});

    var version = std.mem.zeroes(c.nrt_version_t);
    var device: contract.Device = .{ .target = .neuron };
    if (api.nrt_get_version.call(.{ &version, @sizeOf(c.nrt_version_t) })) |_| {
        device.driver_version = try arena.dupe(u8, std.mem.sliceTo(&version.rt_detail, 0));
    } else |_| {}

    var indices: [c.MAX_NEURON_DEVICE_COUNT]c_int = undefined;
    const available = nrt.ndl_available_devices.function(&indices, indices.len);
    const count: usize = @intCast(std.math.clamp(available, 0, indices.len));
    var opened: std.ArrayList(struct { c_int, *c.ndl_device_t }) = try .initCapacity(arena, count);
    for (indices[0..count]) |index| {
        var params: c.ndl_device_init_param_t = .{ .initialize_device = false, .num_dram_regions = 0, .map_hbm = false, .skip_copy_bufs = true };
        // A device that cannot be opened is skipped.
        const handle = (nrt.ndl_open_device.get(.{ index, &params }) catch null) orelse continue;
        opened.appendAssumeCapacity(.{ index, handle });
    }

    // The devices share the cores evenly.
    const cores_per_chip = core_count / @max(1, opened.items.len);
    const chips = try arena.alloc(Chip, opened.items.len);
    const devices = try arena.alloc(contract.Device, chips.len * cores_per_chip);
    const memory = try arena.alloc(?Dir, devices.len);
    @memset(memory, null);
    errdefer closeMemory(memory);

    var path: [64]u8 = undefined;
    for (chips, opened.items, 0..) |*chip, entry, index| {
        const device_id, const handle = entry;
        const first = index * cores_per_chip;
        const cores = memory[first..][0..cores_per_chip];
        chip.* = .{
            .handle = handle,
            .mem_total_bytes = memoryTotal(handle.device_type),
            .memory = cores,
        };
        device.name = null;
        const chip_path = "/sys/devices/virtual/neuron_device/neuron{d}";
        const directory = try std.fmt.bufPrint(&path, chip_path, .{device_id});
        if (Dir.open(io, directory) catch null) |dir| {
            defer dir.close();
            const core_path = "neuron_core{d}/stats/memory_usage/device_mem";
            for (cores, 0..) |*core, core_index| {
                core.* = dir.openDir(try std.fmt.bufPrint(&path, core_path, .{core_index})) catch null;
            }
            var buffer: [128]u8 = undefined;
            if (dir.read("info/architecture/device_name", &buffer) catch null) |name| {
                device.name = try arena.dupe(u8, name);
            }
        }
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

/// Device memory by device type: Inferentia, then Inferentia2 and Trainium, Trainium2, Trainium3.
fn memoryTotal(device_type: u8) ?u64 {
    const gib: u64 = 1 << 30;
    const totals = [_]?u64{ null, 8 * gib, 32 * gib, 96 * gib, 144 * gib };
    return if (device_type < totals.len) totals[device_type] else null;
}

fn closeMemory(memory: []const ?Dir) void {
    for (memory) |entry| if (entry) |dir| dir.close();
}

pub fn close(self: *Neuron) void {
    for (self.chips) |chip| closeMemory(chip.memory);
    self.busy.deinit();
}

const MemoryProbe = struct {
    dir: Dir,

    const Categories = contract.Reply(&.{
        .nc_tensors,       .nc_constants, .nc_model_code, .nc_shared_scratchpad, .nc_nonshared_scratchpad,
        .nc_runtime,       .nc_driver,    .nc_dma_rings,  .nc_collectives,       .nc_notifications,
        .nc_uncategorized,
    });

    pub fn categories(self: *const MemoryProbe) Categories {
        var values: Categories = .{};
        inline for (@typeInfo(Categories).@"struct".field_names) |field| {
            const name = comptime switch (@field(contract.Metric, field)) {
                .nc_shared_scratchpad => "model_shared_scratchpad",
                .nc_runtime => "runtime_memory",
                .nc_driver => "driver_memory",
                else => field["nc_".len..],
            };
            @field(values, field) = self.dir.int(u64, name ++ "/total") catch null;
        }
        return values;
    }
};

pub fn poll(self: *Neuron, scratch: std.mem.Allocator, reading: *source.Reading) !void {
    defer self.busy.sweep();
    var read_any = false;
    var first: usize = 0;
    const nrt = &self.nrt;
    for (self.chips) |chip| {
        const cores = reading.samples[first..][0..chip.memory.len];
        defer first += cores.len;

        // The cores of a chip share its memory: each shows what all of them use.
        const mem_used = memoryInUse(chip.memory);
        for (cores, chip.memory) |*sample, memory| {
            sample.put(.mem_used_bytes, mem_used);
            sample.put(.mem_total_bytes, chip.mem_total_bytes);
            if (memory) |dir| polling.project(sample, &MemoryProbe{ .dir = dir });
        }

        // One scan supplies each core's busy time and the processes that hold memory on it.
        var list: ?[*]c.struct_neuron_app_info = null;
        var count: usize = 0;
        nrt.ndl_get_all_apps_info.call(.{ chip.handle, &list, &count, c.APP_INFO_ALL }) catch continue;
        defer std.c.free(list);
        read_any = true;
        const apps = if (list) |items| items[0..count] else &.{};
        const busy = try scratch.alloc(?native.Growth, cores.len);
        @memset(busy, null);
        for (apps) |app| {
            if (app.pid <= 0) continue;
            const instance = (nrt.nds_open.get(.{ chip.handle, app.pid }) catch null) orelse continue;
            defer _ = nrt.nds_close.function(instance);

            const mem_kib = app.device_mem_size / 1024;
            for (busy, 0..) |*sum, core| {
                const micros = nrt.nds_get_nc_counter.get(.{
                    instance, @intCast(core), c.NDS_NC_COUNTER_TIME_IN_USE,
                }) catch continue;
                const key: Key = .{ .pid = @intCast(app.pid), .core = @intCast(first + core) };
                const nanos = native.units.scale(micros, .micro, .nano);
                if (try self.busy.advance(key, self.io, nanos)) |growth| native.Growth.add(sum, growth);
                if (micros > 0 and mem_kib > 0) try reading.addProcess(.{
                    .pid = key.pid,
                    .device_idx = key.core,
                    .dev_mem_kib = mem_kib,
                });
            }
        }
        for (cores, busy) |*sample, sum| {
            sample.put(.util_percent, if (sum) |growth| growth.percentOfTime() else 0);
        }
    }
    if (!read_any) return error.Unavailable;
    for (reading.processes.items) |*row| {
        row.dev_util_percent = @intCast(reading.samples[row.device_idx].get(.util_percent).?);
    }
}

fn memoryInUse(memory: []const ?Dir) ?u64 {
    var total: u64 = 0;
    for (memory) |dir| total += (dir orelse return null).int(u64, "total") catch return null;
    return total;
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

    var private: Private = undefined;
    inline for (@typeInfo(Private).@"struct".field_names) |field| {
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
        @field(private, field) = .{ .function = @ptrFromInt(address) };
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
