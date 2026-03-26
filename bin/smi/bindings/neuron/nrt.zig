const std = @import("std");
const c = @import("c");
const stdx = @import("stdx");
const DynLib = @import("../dynlib.zig");
const sandbox = @import("../../utils/sandbox.zig");

pub const Error = error{ nrt_error, NrtUnavailable };

const Nrt = @This();
lib: Fns,

const Fns = struct {
    ndl_available_devices: DynLib.Fn("ndl_available_devices"),
    ndl_open_device: DynLib.Fn("ndl_open_device"),
    ndl_close_device: DynLib.Fn("ndl_close_device"),
    ndl_get_all_apps_info: DynLib.Fn("ndl_get_all_apps_info"),
    nds_open: DynLib.Fn("nds_open"),
    nds_close: DynLib.Fn("nds_close"),
    nds_get_nc_counter: DynLib.Fn("nds_get_nc_counter"),
    nrt_get_total_nc_count: DynLib.Fn("nrt_get_total_nc_count"),
};

pub fn init(io: std.Io) Error!Nrt {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = sandbox.path(&path_buf) orelse return error.NrtUnavailable;

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libnrt.so.1" }) catch
        return error.NrtUnavailable;

    return .{ .lib = DynLib.openElf(Fns, io, path, "nrt_init") catch return error.NrtUnavailable };
}

pub fn availableDevices(self: Nrt, buf: []c_int) []c_int {
    const count: usize = @intCast(@max(0, self.lib.ndl_available_devices(buf.ptr, @intCast(buf.len))));
    return buf[0..count];
}

pub fn openDevice(self: Nrt, device_index: c_int) Error!*c.ndl_device_t {
    var dev: ?*c.ndl_device_t = null;
    var t: c.struct_ndl_device_init_param = .{
        .initialize_device = false,
        .map_hbm = false,
        .num_dram_regions = 0,
    };

    if (self.lib.ndl_open_device(device_index, &t, &dev) != 0) return error.nrt_error;
    return dev orelse error.nrt_error;
}

pub fn closeDevice(self: Nrt, dev: *c.ndl_device_t) void {
    _ = self.lib.ndl_close_device(dev);
}

pub const AppInfo = c.struct_neuron_app_info;

pub fn getAllAppsInfo(self: Nrt, dev: *c.ndl_device_t) Error!struct { ptr: ?[*]AppInfo, count: usize } {
    var info: ?[*]AppInfo = null;
    var count: usize = 0;
    if (self.lib.ndl_get_all_apps_info(dev, &info, &count, c.APP_INFO_ALL) != 0) return error.nrt_error;
    return .{ .ptr = info, .count = count };
}

pub fn ndsOpen(self: Nrt, dev: *c.ndl_device_t, pid: c.pid_t) Error!*c.nds_instance_t {
    var inst: ?*c.nds_instance_t = null;
    if (self.lib.nds_open(dev, pid, &inst) != 0) return error.nrt_error;
    return inst orelse error.nrt_error;
}

pub fn ndsClose(self: Nrt, inst: *c.nds_instance_t) void {
    _ = self.lib.nds_close(inst);
}

pub fn getNcCounter(self: Nrt, inst: *c.nds_instance_t, pnc_index: c_int, counter_index: u32) Error!u64 {
    var value: u64 = 0;
    if (self.lib.nds_get_nc_counter(inst, pnc_index, counter_index, &value) != 0) return error.nrt_error;
    return value;
}

pub fn getTotalNcCount(self: Nrt) Error!u32 {
    var count: u32 = 0;
    if (self.lib.nrt_get_total_nc_count(&count) != 0) return error.nrt_error;
    return count;
}

pub fn getHbmSize(dev: *c.ndl_device_t) usize {
    return dev.hbm_size;
}

pub fn getDeviceType(dev: *c.ndl_device_t) c_int {
    return dev.device_type;
}
