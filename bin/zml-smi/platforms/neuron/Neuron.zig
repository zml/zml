const std = @import("std");
const c = @import("c");
const smi = @import("zml-smi");

const Neuron = @This();
const log = std.log.scoped(.@"zml-smi/neuron");

platform: smi.Platform,
io: std.Io,
gpa: std.mem.Allocator,
nrt: Private,
chips: []const Chip,
prev_busy: std.AutoHashMapUnmanaged(Key, std.Io.Duration) = .empty,
curr_busy: std.AutoHashMapUnmanaged(Key, std.Io.Duration) = .empty,
last_poll: ?std.Io.Timestamp = null,
rows: ?[]const smi.ProcessInfo = null,

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes, .deinit = deinit };

const Api = struct {
    nrt_get_total_nc_count: *const @TypeOf(c.nrt_get_total_nc_count),
    nrt_get_version: *const @TypeOf(c.nrt_get_version),
};

/// The functions libnrt does not export, found in its symbol table.
const Private = struct {
    ndl_available_devices: *const @TypeOf(c.ndl_available_devices),
    ndl_open_device: *const @TypeOf(c.ndl_open_device),
    ndl_close_device: *const @TypeOf(c.ndl_close_device),
    ndl_get_all_apps_info: *const @TypeOf(c.ndl_get_all_apps_info),
    nds_open: *const @TypeOf(c.nds_open),
    nds_close: *const @TypeOf(c.nds_close),
    nds_get_nc_counter: *const @TypeOf(c.nds_get_nc_counter),
};

pub fn detect(io: std.Io) bool {
    return smi.sysfs.exists(io, c.NEURON_DEVICE_PREFIX ++ "0");
}

pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !Neuron {
    const api = try open(gpa, io);
    const nrt = try findPrivate(io, api.nrt_get_version);
    var core_count: u32 = 0;
    try check(api.nrt_get_total_nc_count(&core_count));

    var version = std.mem.zeroes(c.nrt_version_t);
    const driver_version = if (api.nrt_get_version(&version, @sizeOf(c.nrt_version_t)) == 0)
        try arena.dupe(u8, std.mem.sliceTo(&version.rt_detail, 0))
    else
        null;

    var indices: [c.MAX_NEURON_DEVICE_COUNT]c_int = undefined;
    const count: usize = @intCast(std.math.clamp(nrt.ndl_available_devices(&indices, indices.len), 0, indices.len));
    var opened: std.ArrayList(struct { c_int, *c.ndl_device_t }) = try .initCapacity(arena, count);
    errdefer for (opened.items) |entry| {
        _ = nrt.ndl_close_device(entry[1]);
    };

    for (indices[0..count]) |index| {
        var params: c.ndl_device_init_param_t = .{ .initialize_device = false, .num_dram_regions = 0, .map_hbm = false, .skip_copy_bufs = true };
        var handle: ?*c.ndl_device_t = null;
        if (nrt.ndl_open_device(index, &params, &handle) != 0) continue;
        opened.appendAssumeCapacity(.{ index, handle orelse continue });
    }

    const cores_per_chip = core_count / @max(1, opened.items.len);
    const chips = try arena.alloc(Chip, opened.items.len);
    var devices: std.ArrayList(smi.Device) = try .initCapacity(arena, chips.len * cores_per_chip);

    var initialized: usize = 0;
    errdefer for (chips[0..initialized]) |chip| {
        for (chip.memory) |memory| if (memory) |dir| dir.close();
    };

    for (chips, opened.items) |*chip, entry| {
        const index, const handle = entry;
        chip.* = try openChip(arena, io, index, handle, cores_per_chip);
        initialized += 1;
        for (chip.memory) |_| devices.appendAssumeCapacity(.{ .target = .neuron, .name = chip.name, .driver_version = driver_version });
    }

    return .{
        .platform = .{ .devices = devices.items, .two_readings = true, .vtable = &vtable },
        .io = io,
        .gpa = gpa,
        .nrt = nrt,
        .chips = chips,
    };
}

fn poll(platform: *smi.Platform, scratch: std.mem.Allocator, samples: []smi.Sample) void {
    const self: *Neuron = @alignCast(@fieldParentPtr("platform", platform));
    self.curr_busy.clearRetainingCapacity();
    defer std.mem.swap(@TypeOf(self.prev_busy), &self.prev_busy, &self.curr_busy);

    const now: std.Io.Timestamp = .now(self.io, .awake);
    const elapsed: ?std.Io.Duration = blk: {
        const last = self.last_poll orelse break :blk null;
        if (now.nanoseconds <= last.nanoseconds) break :blk null;

        break :blk last.durationTo(now);
    };

    self.last_poll = now;
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

        self.scan(scratch, chip.handle, @intCast(first), elapsed, cores, &rows) catch continue;
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

fn deinit(platform: *smi.Platform) void {
    const self: *Neuron = @alignCast(@fieldParentPtr("platform", platform));

    for (self.chips) |chip| {
        for (chip.memory) |memory| if (memory) |dir| dir.close();
        _ = self.nrt.ndl_close_device(chip.handle);
    }

    self.prev_busy.deinit(self.gpa);
    self.curr_busy.deinit(self.gpa);
}

const Chip = struct {
    handle: *c.ndl_device_t,
    name: ?[]const u8 = null,
    mem_total_bytes: ?u64,
    memory: []const ?smi.sysfs.Dir,
};

const Key = struct { pid: u32, core: u16 };

fn open(gpa: std.mem.Allocator, io: std.Io) !Api {
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox = smi.sandbox.path(gpa, io, &buffer) orelse return error.LibraryNotFound;

    var path: [std.Io.Dir.max_path_bytes]u8 = undefined;
    return smi.dynlib.open(Api, &.{try std.fmt.bufPrintSentinel(&path, "{s}/lib/libnrt.so.1", .{sandbox}, 0)});
}

fn openChip(arena: std.mem.Allocator, io: std.Io, index: c_int, handle: *c.ndl_device_t, cores: usize) !Chip {
    const memory = try arena.alloc(?smi.sysfs.Dir, cores);
    @memset(memory, null);
    errdefer for (memory) |dir| if (dir) |d| d.close();

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

fn memoryInUse(memory: []const ?smi.sysfs.Dir) ?u64 {
    var total: u64 = 0;
    for (memory) |dir| {
        total += (dir orelse return null).int(u64, "total") catch return null;
    }

    return total;
}

fn scan(self: *Neuron, scratch: std.mem.Allocator, handle: *c.ndl_device_t, first: u16, elapsed: ?std.Io.Duration, cores: []smi.Sample, rows: *std.ArrayList(smi.ProcessInfo)) !void {
    var list: ?[*]c.struct_neuron_app_info = null;
    var count: usize = 0;
    try check(self.nrt.ndl_get_all_apps_info(handle, &list, &count, c.APP_INFO_ALL));
    defer std.c.free(list);

    const apps: []const c.struct_neuron_app_info = if (list) |items| items[0..count] else &.{};

    const busy = try scratch.alloc(std.Io.Duration, cores.len);
    @memset(busy, .zero);
    for (apps) |app| {
        if (app.pid <= 0) continue;
        var instance: ?*c.nds_instance_t = null;
        if (self.nrt.nds_open(handle, app.pid, &instance) != 0) continue;
        const nds = instance orelse continue;
        defer _ = self.nrt.nds_close(nds);

        for (busy, 0..) |*sum, core| {
            var micros: u64 = 0;
            if (self.nrt.nds_get_nc_counter(nds, @intCast(core), c.NDS_NC_COUNTER_TIME_IN_USE, &micros) != 0) continue;
            const key: Key = .{ .pid = @intCast(app.pid), .core = first + @as(u16, @intCast(core)) };
            const duration: std.Io.Duration = .fromMicroseconds(std.math.cast(i64, micros) orelse continue);
            try self.curr_busy.put(self.gpa, key, duration);
            if (self.prev_busy.get(key)) |prev| {
                if (duration.nanoseconds >= prev.nanoseconds) sum.nanoseconds += duration.nanoseconds - prev.nanoseconds;
            }

            const mem_kib = app.device_mem_size / 1024;
            if (micros > 0 and mem_kib > 0) try rows.append(scratch, .{ .pid = key.pid, .device_idx = key.core, .dev_mem_kib = mem_kib });
        }
    }

    for (cores, busy) |*sample, sum| {
        sample.neuron.util_percent = if (elapsed) |dt| @intCast(@min(@divTrunc(sum.nanoseconds * 100, dt.nanoseconds), 100)) else 0;
    }
}

fn findPrivate(io: std.Io, known: *const anyopaque) !Private {
    var dl_info: DlInfo = undefined;
    if (dladdr(known, &dl_info) == 0) return error.LibraryNotFound;

    const elf_base = @intFromPtr(dl_info.fbase);
    const elf_path = std.mem.span(dl_info.fname);

    const file = try std.Io.Dir.cwd().openFile(io, elf_path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    var map = try file.createMemoryMap(io, .{
        .len = @intCast(stat.size),
        .protection = .{ .read = true },
        .populate = false,
    });
    defer map.destroy(io);

    const elf = std.elf.ElfN;
    const image = map.memory;
    const ehdr: *const elf.Ehdr = @ptrCast(@alignCast(image));
    const shdrs: []const elf.Shdr = @as([*]const elf.Shdr, @ptrCast(@alignCast(image[ehdr.shoff..])))[0..ehdr.shnum];

    const field_names = comptime std.meta.fieldNames(Private);
    var private: Private = undefined;
    var found: [field_names.len]bool = @splat(false);
    var remaining: usize = field_names.len;

    for (shdrs) |sh| {
        if (sh.type != .SYMTAB) continue;

        const strtab_sh = shdrs[sh.link];
        const strtab = image[strtab_sh.offset..][0..strtab_sh.size];
        const syms = std.mem.bytesAsSlice(elf.Sym, image[sh.offset..][0..sh.size]);

        for (syms) |sym| {
            if (sym.value == 0) continue;

            const name = std.mem.sliceTo(strtab[sym.name..], 0);
            inline for (field_names, &found) |field, *seen| {
                if (!seen.* and std.mem.eql(u8, name, field)) {
                    seen.* = true;
                    @field(private, field) = @ptrFromInt(elf_base + sym.value);
                    remaining -= 1;
                }
            }

            if (remaining == 0) return private;
        }
    }

    inline for (field_names, found) |field, seen| {
        if (!seen) log.warn("libnrt has no symbol {s}", .{field});
    }
    return error.MissingSymbol;
}

const DlInfo = extern struct {
    fname: [*:0]const u8,
    fbase: *anyopaque,
    sname: ?[*:0]const u8,
    saddr: ?*anyopaque,
};

extern fn dladdr(address: *const anyopaque, info: *DlInfo) c_int;

fn check(status: i64) error{NrtError}!void {
    if (status != 0) return error.NrtError;
}

test findPrivate {
    const io = std.testing.io;
    const api = try open(std.testing.allocator, io);
    const nrt = try findPrivate(io, api.nrt_get_version);
    var indices: [c.MAX_NEURON_DEVICE_COUNT]c_int = undefined;
    try std.testing.expect(nrt.ndl_available_devices(&indices, indices.len) >= 0);
}
