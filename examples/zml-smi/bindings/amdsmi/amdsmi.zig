const std = @import("std");
const c = @import("c");

pub const Handle = c.amdsmi_processor_handle;

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

fn Fn(comptime name: [:0]const u8) type {
    return *const @TypeOf(@field(c, name));
}

fn sym(comptime name: [:0]const u8) ?Fn(name) {
    if (!lib_loaded) return null;
    return lib.lookup(Fn(name), name);
}

fn call(comptime name: [:0]const u8, args: std.meta.ArgsTuple(@TypeOf(@field(c, name)))) Error!void {
    const f = sym(name) orelse return error.AmdSmiUnavailable;
    try check(@call(.auto, f, args));
}

var gpu_handles: []c.amdsmi_processor_handle = &.{};
var gpu_allocator: std.mem.Allocator = undefined;

var lib: std.DynLib = undefined;
var lib_loaded: bool = false;

fn openLib() !std.DynLib {
    if (std.DynLib.open("libamd_smi.so")) |l| {
        return l;
    } else |_| {}

    if (c.getenv("ROCM_PATH")) |rocm_path| {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrintZ(&buf, "{s}/lib/libamd_smi.so", .{rocm_path}) catch return error.FileNotFound;

        return std.DynLib.open(path);
    }

    return std.DynLib.open("/opt/rocm/lib/libamd_smi.so");
}

pub fn init(allocator: std.mem.Allocator) (Error || error{OutOfMemory})!void {
    lib = openLib() catch return error.AmdSmiUnavailable;
    lib_loaded = true;
    gpu_allocator = allocator;

    try call("amdsmi_init", .{c.AMDSMI_INIT_AMD_GPUS});

    var socket_count: u32 = 0;
    try call("amdsmi_get_socket_handles", .{ &socket_count, null });

    const sockets = try allocator.alloc(c.amdsmi_socket_handle, socket_count);
    defer allocator.free(sockets);
    try call("amdsmi_get_socket_handles", .{ &socket_count, @ptrCast(sockets.ptr) });

    var handle_list: std.ArrayList(c.amdsmi_processor_handle) = .{};
    defer handle_list.deinit(allocator);

    for (sockets[0..socket_count]) |socket| {
        var proc_count: u32 = 0;
        try call("amdsmi_get_processor_handles", .{ socket, &proc_count, null });

        const procs = try allocator.alloc(c.amdsmi_processor_handle, proc_count);
        defer allocator.free(procs);
        try call("amdsmi_get_processor_handles", .{ socket, &proc_count, @ptrCast(procs.ptr) });

        for (procs[0..proc_count]) |proc| {
            try handle_list.append(allocator, proc);
        }
    }

    gpu_handles = try handle_list.toOwnedSlice(allocator);
}

pub fn getHandleByIndex(device_id: u32) Error!Handle {
    if (device_id >= gpu_handles.len) return error.not_found;
    return gpu_handles[device_id];
}

pub fn getDeviceCount() Error!u32 {
    return @intCast(gpu_handles.len);
}

pub const name_buf_len = c.AMDSMI_MAX_STRING_LENGTH;

pub fn getName(handle: Handle, buf: *[c.AMDSMI_MAX_STRING_LENGTH]u8) Error![:0]const u8 {
    var info: c.amdsmi_asic_info_t = undefined;
    try call("amdsmi_get_gpu_asic_info", .{ handle, &info });
    @memcpy(buf, &info.market_name);
    return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(buf)), 0);
}

pub fn getPowerUsage(handle: Handle) Error!u32 {
    var metrics: c.amdsmi_gpu_metrics_t = undefined;
    try call("amdsmi_get_gpu_metrics_info", .{ handle, &metrics });
    const unsupported16 = std.math.maxInt(u16);
    if (metrics.current_socket_power != unsupported16) return metrics.current_socket_power;
    if (metrics.average_socket_power != unsupported16) return metrics.average_socket_power;
    return error.not_supported;
}

pub fn getPowerLimit(handle: Handle) Error!u32 {
    var info: c.amdsmi_power_info_t = undefined;
    try call("amdsmi_get_power_info", .{ handle, &info });
    return info.power_limit;
}

pub fn getPowerCap(handle: Handle) Error!u64 {
    var info: c.amdsmi_power_cap_info_t = undefined;
    try call("amdsmi_get_power_cap_info", .{ handle, @as(u32, 0), &info });
    return info.power_cap;
}

pub fn getTemperature(handle: Handle) Error!i64 {
    var temp: i64 = 0;
    try call("amdsmi_get_temp_metric", .{ handle, c.AMDSMI_TEMPERATURE_TYPE_EDGE, c.AMDSMI_TEMP_CURRENT, &temp });
    return temp;
}

pub fn getTemperatureHotspot(handle: Handle) Error!i64 {
    var temp: i64 = 0;
    try call("amdsmi_get_temp_metric", .{ handle, c.AMDSMI_TEMPERATURE_TYPE_HOTSPOT, c.AMDSMI_TEMP_CURRENT, &temp });
    return temp;
}

pub fn getTemperatureVram(handle: Handle) Error!i64 {
    var temp: i64 = 0;
    try call("amdsmi_get_temp_metric", .{ handle, c.AMDSMI_TEMPERATURE_TYPE_VRAM, c.AMDSMI_TEMP_CURRENT, &temp });
    return temp;
}

pub fn getFanSpeed(handle: Handle) Error!i64 {
    var speed: i64 = 0;
    try call("amdsmi_get_gpu_fan_speed", .{ handle, @as(u32, 0), &speed });
    return speed;
}

pub fn getFanRpms(handle: Handle) Error!i64 {
    var rpms: i64 = 0;
    try call("amdsmi_get_gpu_fan_rpms", .{ handle, @as(u32, 0), &rpms });
    return rpms;
}

pub fn getGpuUtil(handle: Handle) Error!u32 {
    var usage: c.amdsmi_engine_usage_t = undefined;
    try call("amdsmi_get_gpu_activity", .{ handle, &usage });
    return usage.gfx_activity;
}

pub fn getMmUtil(handle: Handle) Error!u32 {
    var usage: c.amdsmi_engine_usage_t = undefined;
    try call("amdsmi_get_gpu_activity", .{ handle, &usage });
    return usage.mm_activity;
}

pub fn getClockGraphics(handle: Handle) Error!u32 {
    var info: c.amdsmi_clk_info_t = undefined;
    try call("amdsmi_get_clock_info", .{ handle, c.AMDSMI_CLK_TYPE_SYS, &info });
    return info.clk;
}

pub fn getMaxClockGraphics(handle: Handle) Error!u32 {
    var info: c.amdsmi_clk_info_t = undefined;
    try call("amdsmi_get_clock_info", .{ handle, c.AMDSMI_CLK_TYPE_SYS, &info });
    return info.max_clk;
}

pub fn getClockMem(handle: Handle) Error!u32 {
    var info: c.amdsmi_clk_info_t = undefined;
    try call("amdsmi_get_clock_info", .{ handle, c.AMDSMI_CLK_TYPE_MEM, &info });
    return info.clk;
}

pub fn getMaxClockMem(handle: Handle) Error!u32 {
    var info: c.amdsmi_clk_info_t = undefined;
    try call("amdsmi_get_clock_info", .{ handle, c.AMDSMI_CLK_TYPE_MEM, &info });
    return info.max_clk;
}

pub fn getClockSoc(handle: Handle) Error!u32 {
    var info: c.amdsmi_clk_info_t = undefined;
    try call("amdsmi_get_clock_info", .{ handle, c.AMDSMI_CLK_TYPE_SOC, &info });
    return info.clk;
}

pub fn getMemTotal(handle: Handle) Error!u64 {
    var total: u64 = 0;
    try call("amdsmi_get_gpu_memory_total", .{ handle, c.AMDSMI_MEM_TYPE_VRAM, &total });
    return total;
}

pub fn getMemUsed(handle: Handle) Error!u64 {
    var used: u64 = 0;
    try call("amdsmi_get_gpu_memory_usage", .{ handle, c.AMDSMI_MEM_TYPE_VRAM, &used });
    return used;
}

pub fn getPcieWidth(handle: Handle) Error!u32 {
    var info: c.amdsmi_pcie_info_t = undefined;
    try call("amdsmi_get_pcie_info", .{ handle, &info });
    return @intCast(info.pcie_metric.pcie_width);
}

pub fn getPcieSpeed(handle: Handle) Error!u32 {
    var info: c.amdsmi_pcie_info_t = undefined;
    try call("amdsmi_get_pcie_info", .{ handle, &info });
    return info.pcie_metric.pcie_speed;
}

pub fn getPcieBandwidth(handle: Handle) Error!u32 {
    var info: c.amdsmi_pcie_info_t = undefined;
    try call("amdsmi_get_pcie_info", .{ handle, &info });
    return info.pcie_metric.pcie_bandwidth;
}

pub fn getPcieLinkGen(handle: Handle) Error!u32 {
    var info: c.amdsmi_pcie_info_t = undefined;
    try call("amdsmi_get_pcie_info", .{ handle, &info });
    return info.pcie_static.pcie_interface_version;
}

pub fn getVoltageGfx(handle: Handle) Error!u64 {
    var info: c.amdsmi_power_info_t = undefined;
    try call("amdsmi_get_power_info", .{ handle, &info });
    return info.gfx_voltage;
}

pub fn getVoltageSoc(handle: Handle) Error!u64 {
    var info: c.amdsmi_power_info_t = undefined;
    try call("amdsmi_get_power_info", .{ handle, &info });
    return info.soc_voltage;
}

pub fn getVoltageMem(handle: Handle) Error!u64 {
    var info: c.amdsmi_power_info_t = undefined;
    try call("amdsmi_get_power_info", .{ handle, &info });
    return info.mem_voltage;
}

pub fn getBdfId(handle: Handle) Error!u64 {
    var bdf_id: u64 = 0;
    try call("amdsmi_get_gpu_bdf_id", .{ handle, &bdf_id });
    return bdf_id;
}

pub const ProcInfo = c.amdsmi_proc_info_t;

pub fn getProcessList(allocator: std.mem.Allocator, handle: Handle) (Error || error{OutOfMemory})![]const ProcInfo {
    var count: u32 = 0;
    try call("amdsmi_get_gpu_process_list", .{ handle, &count, null });
    if (count == 0) return &.{};
    const procs = try allocator.alloc(ProcInfo, count);
    errdefer allocator.free(procs);
    try call("amdsmi_get_gpu_process_list", .{ handle, &count, @ptrCast(procs.ptr) });
    return procs[0..count];
}
