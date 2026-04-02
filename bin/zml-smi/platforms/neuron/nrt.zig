const std = @import("std");
const c = @import("c");
const elf = std.elf;
const stdx = @import("stdx");
const DynLib = @import("zml-smi/dynlib");
const sandbox = @import("zml-smi/sandbox");

const Nrt = @This();

pub const Error = error{ nrt_error, NrtUnavailable };
pub const AppInfo = c.struct_neuron_app_info;
pub const DeviceType = enum(c_int) {
    inf1 = 1,
    inf2_trn1 = 2,
    trn2 = 3,
    trn3 = 4,
    _,
};

lib: Fns,
private_lib: PrivateFns,
handles: []const *c.ndl_device_t,
device_indexes: []const c_int,

const Fns = struct {
    nrt_get_total_nc_count: *const @TypeOf(c.nrt_get_total_nc_count),
    nrt_get_version: *const @TypeOf(c.nrt_get_version),
};

const PrivateFns = struct {
    nds_open: *const @TypeOf(c.nds_open),
    ndl_available_devices: *const @TypeOf(c.ndl_available_devices),
    ndl_open_device: *const @TypeOf(c.ndl_open_device),
    ndl_close_device: *const @TypeOf(c.ndl_close_device),
    ndl_get_all_apps_info: *const @TypeOf(c.ndl_get_all_apps_info),
    nds_close: *const @TypeOf(c.nds_close),
    nds_get_nc_counter: *const @TypeOf(c.nds_get_nc_counter),
};

pub fn init(allocator: std.mem.Allocator, io: std.Io) !Nrt {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = sandbox.path(&path_buf) orelse {
        std.log.err("neuron: sandbox path unavailable", .{});
        return error.NrtUnavailable;
    };

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libnrt.so.1" }) catch {
        std.log.err("neuron: failed to construct libnrt.so.1 path", .{});
        return error.NrtUnavailable;
    };

    const dl_handle = std.c.dlopen(path, .{ .LAZY = true, .GLOBAL = true, .NODELETE = true }) orelse {
        if (std.c.dlerror()) |err| std.log.err("neuron: dlopen: {s}", .{err});
        return error.NrtUnavailable;
    };
    var dynlib: std.DynLib = .{ .inner = .{ .handle = dl_handle } };

    const fns = DynLib.lookupStruct(&dynlib, Fns) catch return error.NrtUnavailable;
    const private_fns = resolveFromElf(dl_handle, io, "nrt_init") catch |err| {
        std.log.err("neuron: ELF resolution failed: {s}", .{@errorName(err)});
        return error.NrtUnavailable;
    };

    var dev_index_buf: [c.MAX_NEURON_DEVICE_COUNT]c_int = undefined;
    const count: usize = @intCast(@max(0, private_fns.ndl_available_devices(&dev_index_buf, c.MAX_NEURON_DEVICE_COUNT)));

    var handle_list: std.ArrayList(*c.ndl_device_t) = .{};
    var index_list: std.ArrayList(c_int) = .{};

    for (dev_index_buf[0..count]) |device_idx| {
        var dev: ?*c.ndl_device_t = null;
        var t: c.struct_ndl_device_init_param = .{
            .initialize_device = false,
            .map_hbm = false,
            .num_dram_regions = 0,
        };

        if (private_fns.ndl_open_device(device_idx, &t, &dev) == 0) {
            if (dev) |d| {
                errdefer _ = private_fns.ndl_close_device(d);
                try handle_list.append(allocator, d);
                try index_list.append(allocator, device_idx);
            }
        }
    }

    return .{
        .lib = fns,
        .private_lib = private_fns,
        .handles = try handle_list.toOwnedSlice(allocator),
        .device_indexes = try index_list.toOwnedSlice(allocator),
    };
}

pub fn deviceCount(self: Nrt) u32 {
    return @intCast(self.handles.len);
}

pub fn allAppsInfo(self: Nrt, dev: *c.ndl_device_t) Error!struct { ptr: ?[*]AppInfo, count: usize } {
    var info: ?[*]AppInfo = null;
    var count: usize = 0;
    if (self.private_lib.ndl_get_all_apps_info(dev, &info, &count, c.APP_INFO_ALL) != 0) {
        return error.nrt_error;
    }
    return .{ .ptr = info, .count = count };
}

pub fn ndsOpen(self: Nrt, dev: *c.ndl_device_t, pid: c.pid_t) Error!*c.nds_instance_t {
    var inst: ?*c.nds_instance_t = null;
    if (self.private_lib.nds_open(dev, pid, &inst) != 0) {
        return error.nrt_error;
    }
    return inst orelse error.nrt_error;
}

pub fn ndsClose(self: Nrt, inst: *c.nds_instance_t) void {
    _ = self.private_lib.nds_close(inst);
}

pub fn ncCounter(self: Nrt, inst: *c.nds_instance_t, pnc_index: c_int, counter_index: u32) Error!u64 {
    var value: u64 = 0;
    if (self.private_lib.nds_get_nc_counter(inst, pnc_index, counter_index, &value) != 0) {
        return error.nrt_error;
    }
    return value;
}

pub const version_buf_len = c.RT_VERSION_DETAIL_LEN;

pub fn version(self: Nrt, buf: *[version_buf_len]u8) Error![:0]const u8 {
    var ver: c.nrt_version_t = std.mem.zeroes(c.nrt_version_t);
    if (self.lib.nrt_get_version(&ver, @sizeOf(c.nrt_version_t)) != 0) {
        return error.nrt_error;
    }
    @memcpy(buf, &ver.rt_detail);
    return std.mem.span(@as([*c]const u8, @ptrCast(buf)));
}

pub fn totalNcCount(self: Nrt) Error!u32 {
    var count: u32 = 0;
    if (self.lib.nrt_get_total_nc_count(&count) != 0) {
        return error.nrt_error;
    }
    return count;
}

pub fn hbmSize(dev: *c.ndl_device_t) usize {
    return dev.hbm_size;
}

pub fn deviceType(dev: *c.ndl_device_t) DeviceType {
    return @enumFromInt(dev.device_type);
}

fn resolveFromElf(handle: *anyopaque, io: std.Io, comptime known_sym: [:0]const u8) !PrivateFns {
    const fields = std.meta.fields(PrivateFns);

    const sym_addr = std.c.dlsym(handle, known_sym) orelse
        return error.SymbolNotFound;

    var dl_info: DlInfo = undefined;
    if (dladdr(sym_addr, &dl_info) == 0) return error.DladdrFailed;

    const elf_base = @intFromPtr(dl_info.dli_fbase);
    const elf_path = std.mem.span(dl_info.dli_fname);

    const file = try std.Io.Dir.openFileAbsolute(io, elf_path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    const mmap = try std.posix.mmap(
        null,
        @intCast(stat.size),
        .{ .READ = true },
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    defer std.posix.munmap(mmap);

    var result: PrivateFns = undefined;
    var remaining: usize = fields.len;

    const ehdr: *const elf.ElfN.Ehdr = @ptrCast(@alignCast(mmap));
    const shdrs: []const elf.ElfN.Shdr = @as([*]const elf.ElfN.Shdr, @ptrCast(@alignCast(mmap[ehdr.shoff..])))[0..ehdr.shnum];

    for (shdrs) |sh| {
        if (sh.type != .SYMTAB) {
            continue;
        }

        const strtab_sh = shdrs[sh.link];
        const strtab: []const u8 = mmap[strtab_sh.offset..][0..strtab_sh.size];

        for (std.mem.bytesAsSlice(elf.ElfN.Sym, mmap[sh.offset..][0..sh.size])) |s| {
            if (s.value == 0) {
                continue;
            }

            const sym_name = std.mem.span(@as([*c]const u8, @ptrCast(strtab[@as(usize, s.name)..])));

            inline for (fields) |field| {
                if (std.mem.eql(u8, sym_name, field.name)) {
                    @field(result, field.name) = @ptrFromInt(elf_base + @as(usize, s.value));
                    remaining -= 1;
                }
            }

            if (remaining == 0) {
                return result;
            }
        }
    }

    if (remaining > 0) {
        return error.SymbolResolutionFailed;
    }
    return result;
}

const DlInfo = extern struct {
    dli_fname: [*:0]const u8,
    dli_fbase: *anyopaque,
    dli_sname: [*c]const u8,
    dli_saddr: ?*anyopaque,
};

extern fn dladdr(addr: *anyopaque, info: *DlInfo) c_int;
