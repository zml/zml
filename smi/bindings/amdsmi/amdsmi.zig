const std = @import("std");
const c = @import("c");
const stdx = @import("stdx");
const DynLib = @import("../dynlib.zig");
const sandbox = @import("../../sandbox.zig");

pub const Handle = c.amdsmi_processor_handle;
pub const ProcInfo = c.amdsmi_proc_info_t;

pub const Error = ReturnError || error{AmdSmiUnavailable};

pub const ReturnError = error{
    inval,
    not_supported,
    not_yet_implemented,
    fail_load_module,
    fail_load_symbol,
    drm_error,
    api_failed,
    timeout,
    retry,
    no_perm,
    interrupt,
    io_error,
    address_fault,
    file_error,
    out_of_resources,
    internal_exception,
    input_out_of_bounds,
    init_error,
    refcount_overflow,
    directory_not_found,
    busy,
    not_found,
    not_init,
    no_slot,
    driver_not_loaded,
    no_data,
    insufficient_size,
    unexpected_size,
    unexpected_data,
    unknown_error,
};

fn check(ret: c.amdsmi_status_t) Error!void {
    if (ret == c.AMDSMI_STATUS_SUCCESS) return;

    return switch (ret) {
        c.AMDSMI_STATUS_INVAL => error.inval,
        c.AMDSMI_STATUS_NOT_SUPPORTED => error.not_supported,
        c.AMDSMI_STATUS_NOT_YET_IMPLEMENTED => error.not_yet_implemented,
        c.AMDSMI_STATUS_FAIL_LOAD_MODULE => error.fail_load_module,
        c.AMDSMI_STATUS_FAIL_LOAD_SYMBOL => error.fail_load_symbol,
        c.AMDSMI_STATUS_DRM_ERROR => error.drm_error,
        c.AMDSMI_STATUS_API_FAILED => error.api_failed,
        c.AMDSMI_STATUS_TIMEOUT => error.timeout,
        c.AMDSMI_STATUS_RETRY => error.retry,
        c.AMDSMI_STATUS_NO_PERM => error.no_perm,
        c.AMDSMI_STATUS_INTERRUPT => error.interrupt,
        c.AMDSMI_STATUS_IO => error.io_error,
        c.AMDSMI_STATUS_ADDRESS_FAULT => error.address_fault,
        c.AMDSMI_STATUS_FILE_ERROR => error.file_error,
        c.AMDSMI_STATUS_OUT_OF_RESOURCES => error.out_of_resources,
        c.AMDSMI_STATUS_INTERNAL_EXCEPTION => error.internal_exception,
        c.AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS => error.input_out_of_bounds,
        c.AMDSMI_STATUS_INIT_ERROR => error.init_error,
        c.AMDSMI_STATUS_REFCOUNT_OVERFLOW => error.refcount_overflow,
        c.AMDSMI_STATUS_DIRECTORY_NOT_FOUND => error.directory_not_found,
        c.AMDSMI_STATUS_BUSY => error.busy,
        c.AMDSMI_STATUS_NOT_FOUND => error.not_found,
        c.AMDSMI_STATUS_NOT_INIT => error.not_init,
        c.AMDSMI_STATUS_NO_SLOT => error.no_slot,
        c.AMDSMI_STATUS_DRIVER_NOT_LOADED => error.driver_not_loaded,
        c.AMDSMI_STATUS_NO_DATA => error.no_data,
        c.AMDSMI_STATUS_INSUFFICIENT_SIZE => error.insufficient_size,
        c.AMDSMI_STATUS_UNEXPECTED_SIZE => error.unexpected_size,
        c.AMDSMI_STATUS_UNEXPECTED_DATA => error.unexpected_data,
        else => error.unknown_error,
    };
}

const AmdSmi = @This();
lib: Fns,
gpu_handles: []c.amdsmi_processor_handle,

const Fns = struct {
    amdsmi_init: DynLib.Fn("amdsmi_init"),
    amdsmi_get_socket_handles: DynLib.Fn("amdsmi_get_socket_handles"),
    amdsmi_get_processor_handles: DynLib.Fn("amdsmi_get_processor_handles"),
    amdsmi_get_gpu_asic_info: DynLib.Fn("amdsmi_get_gpu_asic_info"),
    amdsmi_get_gpu_metrics_info: DynLib.Fn("amdsmi_get_gpu_metrics_info"),
    amdsmi_get_power_info: DynLib.Fn("amdsmi_get_power_info"),
    amdsmi_get_temp_metric: DynLib.Fn("amdsmi_get_temp_metric"),
    amdsmi_get_gpu_fan_speed: DynLib.Fn("amdsmi_get_gpu_fan_speed"),
    amdsmi_get_gpu_activity: DynLib.Fn("amdsmi_get_gpu_activity"),
    amdsmi_get_clock_info: DynLib.Fn("amdsmi_get_clock_info"),
    amdsmi_get_gpu_memory_total: DynLib.Fn("amdsmi_get_gpu_memory_total"),
    amdsmi_get_gpu_memory_usage: DynLib.Fn("amdsmi_get_gpu_memory_usage"),
    amdsmi_get_pcie_info: DynLib.Fn("amdsmi_get_pcie_info"),
    amdsmi_get_gpu_bdf_id: DynLib.Fn("amdsmi_get_gpu_bdf_id"),
    amdsmi_get_gpu_process_list: DynLib.Fn("amdsmi_get_gpu_process_list"),
};

pub fn init(allocator: std.mem.Allocator) !AmdSmi {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = sandbox.path(&path_buf) orelse return error.AmdSmiUnavailable;

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libamd_smi.so.26" });

    const fns = DynLib.open(Fns, path) orelse return error.AmdSmiUnavailable;

    try check(fns.amdsmi_init(c.AMDSMI_INIT_AMD_GPUS));

    var socket_count: u32 = 0;
    try check(fns.amdsmi_get_socket_handles(&socket_count, null));

    const sockets = try allocator.alloc(c.amdsmi_socket_handle, socket_count);
    defer allocator.free(sockets);
    try check(fns.amdsmi_get_socket_handles(&socket_count, @ptrCast(sockets.ptr)));

    var handle_list: std.ArrayList(c.amdsmi_processor_handle) = .{};
    defer handle_list.deinit(allocator);

    for (sockets[0..socket_count]) |socket| {
        var proc_count: u32 = 0;
        try check(fns.amdsmi_get_processor_handles(socket, &proc_count, null));

        const procs = try allocator.alloc(c.amdsmi_processor_handle, proc_count);
        defer allocator.free(procs);
        try check(fns.amdsmi_get_processor_handles(socket, &proc_count, @ptrCast(procs.ptr)));

        for (procs[0..proc_count]) |proc| {
            try handle_list.append(allocator, proc);
        }
    }

    return .{ .lib = fns, .gpu_handles = try handle_list.toOwnedSlice(allocator) };
}

pub fn getHandleByIndex(self: AmdSmi, device_id: u32) Error!Handle {
    if (device_id >= self.gpu_handles.len) return error.not_found;
    return self.gpu_handles[device_id];
}

pub fn getDeviceCount(self: AmdSmi) u32 {
    return @intCast(self.gpu_handles.len);
}

pub const name_buf_len = c.AMDSMI_MAX_STRING_LENGTH;

pub fn getName(self: AmdSmi, handle: Handle, buf: *[c.AMDSMI_MAX_STRING_LENGTH]u8) Error![:0]const u8 {
    var info: c.amdsmi_asic_info_t = undefined;
    try check(self.lib.amdsmi_get_gpu_asic_info(handle, &info));
    @memcpy(buf, &info.market_name);
    return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(buf)), 0);
}

pub fn getPowerUsage(self: AmdSmi, handle: Handle) Error!u32 {
    var metrics: c.amdsmi_gpu_metrics_t = undefined;
    try check(self.lib.amdsmi_get_gpu_metrics_info(handle, &metrics));
    const unsupported16 = std.math.maxInt(u16);
    if (metrics.current_socket_power != unsupported16) return metrics.current_socket_power;
    if (metrics.average_socket_power != unsupported16) return metrics.average_socket_power;
    return error.not_supported;
}

pub fn getPowerLimit(self: AmdSmi, handle: Handle) Error!u32 {
    var info: c.amdsmi_power_info_t = undefined;
    try check(self.lib.amdsmi_get_power_info(handle, &info));
    return info.power_limit;
}

pub fn getTemperature(self: AmdSmi, handle: Handle) Error!i64 {
    var temp: i64 = 0;
    try check(self.lib.amdsmi_get_temp_metric(handle, c.AMDSMI_TEMPERATURE_TYPE_EDGE, c.AMDSMI_TEMP_CURRENT, &temp));
    return temp;
}

pub fn getFanSpeed(self: AmdSmi, handle: Handle) Error!i64 {
    var speed: i64 = 0;
    try check(self.lib.amdsmi_get_gpu_fan_speed(handle, @as(u32, 0), &speed));
    return speed;
}

pub fn getGpuUtil(self: AmdSmi, handle: Handle) Error!u32 {
    var usage: c.amdsmi_engine_usage_t = undefined;
    try check(self.lib.amdsmi_get_gpu_activity(handle, &usage));
    return usage.gfx_activity;
}

pub fn getClockGraphics(self: AmdSmi, handle: Handle) Error!u32 {
    var info: c.amdsmi_clk_info_t = undefined;
    try check(self.lib.amdsmi_get_clock_info(handle, c.AMDSMI_CLK_TYPE_SYS, &info));
    return info.clk;
}

pub fn getMaxClockGraphics(self: AmdSmi, handle: Handle) Error!u32 {
    var info: c.amdsmi_clk_info_t = undefined;
    try check(self.lib.amdsmi_get_clock_info(handle, c.AMDSMI_CLK_TYPE_SYS, &info));
    return info.max_clk;
}

pub fn getClockMem(self: AmdSmi, handle: Handle) Error!u32 {
    var info: c.amdsmi_clk_info_t = undefined;
    try check(self.lib.amdsmi_get_clock_info(handle, c.AMDSMI_CLK_TYPE_MEM, &info));
    return info.clk;
}

pub fn getMaxClockMem(self: AmdSmi, handle: Handle) Error!u32 {
    var info: c.amdsmi_clk_info_t = undefined;
    try check(self.lib.amdsmi_get_clock_info(handle, c.AMDSMI_CLK_TYPE_MEM, &info));
    return info.max_clk;
}

pub fn getClockSoc(self: AmdSmi, handle: Handle) Error!u32 {
    var info: c.amdsmi_clk_info_t = undefined;
    try check(self.lib.amdsmi_get_clock_info(handle, c.AMDSMI_CLK_TYPE_SOC, &info));
    return info.clk;
}

pub fn getMemTotal(self: AmdSmi, handle: Handle) Error!u64 {
    var total: u64 = 0;
    try check(self.lib.amdsmi_get_gpu_memory_total(handle, c.AMDSMI_MEM_TYPE_VRAM, &total));
    return total;
}

pub fn getMemUsed(self: AmdSmi, handle: Handle) Error!u64 {
    var used: u64 = 0;
    try check(self.lib.amdsmi_get_gpu_memory_usage(handle, c.AMDSMI_MEM_TYPE_VRAM, &used));
    return used;
}

pub fn getPcieWidth(self: AmdSmi, handle: Handle) Error!u32 {
    var info: c.amdsmi_pcie_info_t = undefined;
    try check(self.lib.amdsmi_get_pcie_info(handle, &info));
    return @intCast(info.pcie_metric.pcie_width);
}

pub fn getPcieBandwidth(self: AmdSmi, handle: Handle) Error!u32 {
    var info: c.amdsmi_pcie_info_t = undefined;
    try check(self.lib.amdsmi_get_pcie_info(handle, &info));
    return info.pcie_metric.pcie_bandwidth;
}

pub fn getPcieLinkGen(self: AmdSmi, handle: Handle) Error!u32 {
    var info: c.amdsmi_pcie_info_t = undefined;
    try check(self.lib.amdsmi_get_pcie_info(handle, &info));
    return info.pcie_static.pcie_interface_version;
}

pub fn getBdfId(self: AmdSmi, handle: Handle) Error!u64 {
    var bdf_id: u64 = 0;
    try check(self.lib.amdsmi_get_gpu_bdf_id(handle, &bdf_id));
    return bdf_id;
}

pub fn getProcessList(self: AmdSmi, allocator: std.mem.Allocator, handle: Handle) (Error || error{OutOfMemory})![]const ProcInfo {
    var count: u32 = 0;
    try check(self.lib.amdsmi_get_gpu_process_list(handle, &count, null));
    if (count == 0) return &.{};
    const procs = try allocator.alloc(ProcInfo, count);
    errdefer allocator.free(procs);
    try check(self.lib.amdsmi_get_gpu_process_list(handle, &count, @ptrCast(procs.ptr)));
    return procs;
}
