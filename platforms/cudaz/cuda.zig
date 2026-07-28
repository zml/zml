const std = @import("std");

const log = std.log.scoped(.@"zml/platforms/cudaz/cuda");

pub const Error = error{
    DriverUnavailable,
    MissingDriverSymbol,
    InitializationFailed,
    DeviceUnavailable,
    ContextFailed,
    StreamFailed,
    HostRegistrationFailed,
    DeviceAllocationFailed,
    HostToDeviceCopyFailed,
    ModuleLoadFailed,
    FunctionLookupFailed,
};

pub const DeviceHandle = c_int;
pub const Context = ?*anyopaque;
pub const StreamHandle = ?*anyopaque;
pub const ModuleHandle = ?*anyopaque;
pub const FunctionHandle = ?*anyopaque;
pub const DevicePtr = u64;

const Result = c_int;
const success: Result = 0;
const stream_non_blocking: c_uint = 1;

const Api = struct {
    library: std.DynLib,

    init: *const fn (c_uint) callconv(.c) Result,
    deviceGet: *const fn (*DeviceHandle, c_int) callconv(.c) Result,
    devicePrimaryCtxRetain: *const fn (*Context, DeviceHandle) callconv(.c) Result,
    devicePrimaryCtxRelease: *const fn (DeviceHandle) callconv(.c) Result,
    ctxSetCurrent: *const fn (Context) callconv(.c) Result,
    streamCreate: *const fn (*StreamHandle, c_uint) callconv(.c) Result,
    streamDestroy: *const fn (StreamHandle) callconv(.c) Result,
    streamSynchronize: *const fn (StreamHandle) callconv(.c) Result,
    memHostRegister: *const fn (?*anyopaque, usize, c_uint) callconv(.c) Result,
    memHostUnregister: *const fn (?*anyopaque) callconv(.c) Result,
    memAlloc: *const fn (*DevicePtr, usize) callconv(.c) Result,
    memFree: *const fn (DevicePtr) callconv(.c) Result,
    memcpyHtoDAsync: *const fn (DevicePtr, ?*const anyopaque, usize, StreamHandle) callconv(.c) Result,
    moduleLoadData: *const fn (*ModuleHandle, ?*const anyopaque) callconv(.c) Result,
    moduleUnload: *const fn (ModuleHandle) callconv(.c) Result,
    moduleGetFunction: *const fn (*FunctionHandle, ModuleHandle, [*:0]const u8) callconv(.c) Result,

    fn load() Error!Api {
        var library = std.DynLib.open("libcuda.so.1") catch return error.DriverUnavailable;
        errdefer library.close();

        return .{
            .library = library,
            .init = try lookup(*const fn (c_uint) callconv(.c) Result, &library, "cuInit"),
            .deviceGet = try lookup(*const fn (*DeviceHandle, c_int) callconv(.c) Result, &library, "cuDeviceGet"),
            .devicePrimaryCtxRetain = try lookup(*const fn (*Context, DeviceHandle) callconv(.c) Result, &library, "cuDevicePrimaryCtxRetain"),
            .devicePrimaryCtxRelease = try lookup(*const fn (DeviceHandle) callconv(.c) Result, &library, "cuDevicePrimaryCtxRelease_v2"),
            .ctxSetCurrent = try lookup(*const fn (Context) callconv(.c) Result, &library, "cuCtxSetCurrent"),
            .streamCreate = try lookup(*const fn (*StreamHandle, c_uint) callconv(.c) Result, &library, "cuStreamCreate"),
            .streamDestroy = try lookup(*const fn (StreamHandle) callconv(.c) Result, &library, "cuStreamDestroy_v2"),
            .streamSynchronize = try lookup(*const fn (StreamHandle) callconv(.c) Result, &library, "cuStreamSynchronize"),
            .memHostRegister = try lookup(*const fn (?*anyopaque, usize, c_uint) callconv(.c) Result, &library, "cuMemHostRegister_v2"),
            .memHostUnregister = try lookup(*const fn (?*anyopaque) callconv(.c) Result, &library, "cuMemHostUnregister"),
            .memAlloc = try lookup(*const fn (*DevicePtr, usize) callconv(.c) Result, &library, "cuMemAlloc_v2"),
            .memFree = try lookup(*const fn (DevicePtr) callconv(.c) Result, &library, "cuMemFree_v2"),
            .memcpyHtoDAsync = try lookup(*const fn (DevicePtr, ?*const anyopaque, usize, StreamHandle) callconv(.c) Result, &library, "cuMemcpyHtoDAsync_v2"),
            .moduleLoadData = try lookup(*const fn (*ModuleHandle, ?*const anyopaque) callconv(.c) Result, &library, "cuModuleLoadData"),
            .moduleUnload = try lookup(*const fn (ModuleHandle) callconv(.c) Result, &library, "cuModuleUnload"),
            .moduleGetFunction = try lookup(*const fn (*FunctionHandle, ModuleHandle, [*:0]const u8) callconv(.c) Result, &library, "cuModuleGetFunction"),
        };
    }

    fn lookup(comptime T: type, library: *std.DynLib, name: [:0]const u8) Error!T {
        return library.lookup(T, name) orelse {
            log.err("CUDA driver is missing symbol {s}", .{name});
            return error.MissingDriverSymbol;
        };
    }

    fn check(result: Result, operation: []const u8, err: Error) Error!void {
        if (result == success) return;
        log.err("{s} failed with CUDA result {d}", .{ operation, result });
        return err;
    }
};

pub const Device = struct {
    handle: DeviceHandle,
    context: Context,
};

pub const Stream = struct {
    handle: StreamHandle,
};

pub const Allocation = struct {
    ptr: DevicePtr,
    size: usize,
};

pub const Kernel = struct {
    module: ModuleHandle,
    function: FunctionHandle,
};

pub const Client = struct {
    api: Api,
    device: Device,
    stream: Stream,

    pub fn init() Error!Client {
        var api = try Api.load();
        errdefer api.library.close();

        try Api.check(api.init(0), "cuInit", error.InitializationFailed);

        var device_handle: DeviceHandle = undefined;
        try Api.check(api.deviceGet(&device_handle, 0), "cuDeviceGet", error.DeviceUnavailable);

        var context: Context = null;
        try Api.check(api.devicePrimaryCtxRetain(&context, device_handle), "cuDevicePrimaryCtxRetain", error.ContextFailed);
        errdefer _ = api.devicePrimaryCtxRelease(device_handle);

        try Api.check(api.ctxSetCurrent(context), "cuCtxSetCurrent", error.ContextFailed);

        var stream_handle: StreamHandle = null;
        try Api.check(api.streamCreate(&stream_handle, stream_non_blocking), "cuStreamCreate", error.StreamFailed);
        errdefer _ = api.streamDestroy(stream_handle);

        return .{
            .api = api,
            .device = .{
                .handle = device_handle,
                .context = context,
            },
            .stream = .{ .handle = stream_handle },
        };
    }

    pub fn deinit(self: *Client) void {
        _ = self.api.ctxSetCurrent(self.device.context);
        _ = self.api.streamDestroy(self.stream.handle);
        _ = self.api.devicePrimaryCtxRelease(self.device.handle);
        self.api.library.close();
        self.* = undefined;
    }

    pub fn allocate(self: *Client, size: usize) Error!Allocation {
        if (size == 0) return .{ .ptr = 0, .size = 0 };
        try self.setCurrent();

        var device_ptr: DevicePtr = 0;
        try Api.check(self.api.memAlloc(&device_ptr, size), "cuMemAlloc_v2", error.DeviceAllocationFailed);
        return .{
            .ptr = device_ptr,
            .size = size,
        };
    }

    pub fn registerHostMemory(self: *Client, data: []u8) Error!void {
        try self.setCurrent();
        try Api.check(
            self.api.memHostRegister(data.ptr, data.len, 0),
            "cuMemHostRegister_v2",
            error.HostRegistrationFailed,
        );
    }

    pub fn unregisterHostMemory(self: *Client, data: *anyopaque) Error!void {
        try self.setCurrent();
        try Api.check(
            self.api.memHostUnregister(data),
            "cuMemHostUnregister",
            error.HostRegistrationFailed,
        );
    }

    pub fn free(self: *Client, allocation: Allocation) void {
        if (allocation.size == 0) return;
        self.setCurrent() catch return;
        const result = self.api.memFree(allocation.ptr);
        if (result != success) {
            log.err("cuMemFree_v2 failed with CUDA result {d}", .{result});
        }
    }

    pub fn copyHostToDevice(self: *Client, allocation: Allocation, offset: usize, data: []const u8) Error!void {
        if (offset > allocation.size or data.len > allocation.size - offset) {
            return error.HostToDeviceCopyFailed;
        }
        if (data.len == 0) return;
        try self.setCurrent();

        const destination = std.math.add(DevicePtr, allocation.ptr, offset) catch return error.HostToDeviceCopyFailed;
        try Api.check(
            self.api.memcpyHtoDAsync(destination, data.ptr, data.len, self.stream.handle),
            "cuMemcpyHtoDAsync_v2",
            error.HostToDeviceCopyFailed,
        );
        try Api.check(
            self.api.streamSynchronize(self.stream.handle),
            "cuStreamSynchronize",
            error.HostToDeviceCopyFailed,
        );
    }

    pub fn loadKernel(self: *Client, ptx: [:0]const u8, name: [:0]const u8) Error!Kernel {
        try self.setCurrent();

        var module: ModuleHandle = null;
        try Api.check(
            self.api.moduleLoadData(&module, ptx.ptr),
            "cuModuleLoadData",
            error.ModuleLoadFailed,
        );
        errdefer _ = self.api.moduleUnload(module);

        var function: FunctionHandle = null;
        try Api.check(
            self.api.moduleGetFunction(&function, module, name.ptr),
            "cuModuleGetFunction",
            error.FunctionLookupFailed,
        );
        return .{
            .module = module,
            .function = function,
        };
    }

    pub fn unloadKernel(self: *Client, kernel: Kernel) void {
        self.setCurrent() catch return;
        const result = self.api.moduleUnload(kernel.module);
        if (result != success) {
            log.err("cuModuleUnload failed with CUDA result {d}", .{result});
        }
    }

    fn setCurrent(self: *Client) Error!void {
        try Api.check(self.api.ctxSetCurrent(self.device.context), "cuCtxSetCurrent", error.ContextFailed);
    }
};
