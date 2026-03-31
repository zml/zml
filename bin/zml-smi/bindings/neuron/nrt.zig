const std = @import("std");
const c = @import("c");
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
handles: []const *c.ndl_device_t,
device_indexes: []const c_int,

const Fns = struct {
    ndl_available_devices: DynLib.Fn(c,"ndl_available_devices"),
    ndl_open_device: DynLib.Fn(c,"ndl_open_device"),
    ndl_close_device: DynLib.Fn(c,"ndl_close_device"),
    ndl_get_all_apps_info: DynLib.Fn(c,"ndl_get_all_apps_info"),
    nds_open: DynLib.Fn(c,"nds_open"),
    nds_close: DynLib.Fn(c,"nds_close"),
    nds_get_nc_counter: DynLib.Fn(c,"nds_get_nc_counter"),
    nrt_get_total_nc_count: DynLib.Fn(c,"nrt_get_total_nc_count"),
};

pub fn init(allocator: std.mem.Allocator, io: std.Io) Error!Nrt {
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

    const lib = DynLib.openElf(Fns, io, path, "nrt_init") catch |err| {
        std.log.err("neuron: failed to open {s}: {s}", .{ path, @errorName(err) });
        return error.NrtUnavailable;
    };

    var dev_index_buf: [c.MAX_NEURON_DEVICE_COUNT]c_int = undefined;
    const count: usize = @intCast(@max(0, lib.ndl_available_devices(&dev_index_buf, c.MAX_NEURON_DEVICE_COUNT)));

    var handle_list: std.ArrayList(*c.ndl_device_t) = .{};
    var index_list: std.ArrayList(c_int) = .{};

    for (dev_index_buf[0..count]) |device_idx| {
        var dev: ?*c.ndl_device_t = null;
        var t: c.struct_ndl_device_init_param = .{
            .initialize_device = false,
            .map_hbm = false,
            .num_dram_regions = 0,
        };

        if (lib.ndl_open_device(device_idx, &t, &dev) == 0) {
            if (dev) |d| {
                handle_list.append(allocator, d) catch continue;
                index_list.append(allocator, device_idx) catch continue;
            }
        }
    }

    return .{
        .lib = lib,
        .handles = handle_list.toOwnedSlice(allocator) catch return error.NrtUnavailable,
        .device_indexes = index_list.toOwnedSlice(allocator) catch return error.NrtUnavailable,
    };
}

pub fn deviceCount(self: Nrt) u32 {
    return @intCast(self.handles.len);
}

pub fn allAppsInfo(self: Nrt, dev: *c.ndl_device_t) Error!struct { ptr: ?[*]AppInfo, count: usize } {
    var info: ?[*]AppInfo = null;
    var count: usize = 0;
    if (self.lib.ndl_get_all_apps_info(dev, &info, &count, c.APP_INFO_ALL) != 0) {
        return error.nrt_error;
    }
    return .{ .ptr = info, .count = count };
}

pub fn ndsOpen(self: Nrt, dev: *c.ndl_device_t, pid: c.pid_t) Error!*c.nds_instance_t {
    var inst: ?*c.nds_instance_t = null;
    if (self.lib.nds_open(dev, pid, &inst) != 0) {
        return error.nrt_error;
    }
    return inst orelse error.nrt_error;
}

pub fn ndsClose(self: Nrt, inst: *c.nds_instance_t) void {
    _ = self.lib.nds_close(inst);
}

pub fn ncCounter(self: Nrt, inst: *c.nds_instance_t, pnc_index: c_int, counter_index: u32) Error!u64 {
    var value: u64 = 0;
    if (self.lib.nds_get_nc_counter(inst, pnc_index, counter_index, &value) != 0) {
        return error.nrt_error;
    }
    return value;
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
